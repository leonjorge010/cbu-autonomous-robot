"""Apply the wait and mixed-pedestrian changes on the lab checkout."""
from pathlib import Path

ROOT = Path('/lab/home/jduran/cbu-autonomous-robot')


def sub(path, old, new, label):
    file = ROOT / path
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label} found {count}')
    file.write_text(text.replace(old, new, 1))
    print(label)


sub('src/cash_rl/cash_rl/raycast_env.py',
    'PEDESTRIAN_SPEED_RANGE = (0.5, 0.9)\n',
    'PEDESTRIAN_SPEED_RANGE = (0.5, 0.9)\n'
    'CROSS_SPEED_RANGE = (1.0, 1.4)\n',
    'speed')

sub('src/cash_rl/cash_rl/raycast_env.py',
    '                 pedestrians_yield=False, lane_penalty=True, encounter_prob=0.0,\n'
    '                 subgoal_lookahead=None, subgoal_offset=0.0, lane_hold=LANE_HOLD_RANGE):',
    '                 pedestrians_yield=False, lane_penalty=True, encounter_prob=0.0,\n'
    '                 yield_prob=0.0, cross_prob=0.0,\n'
    '                 subgoal_lookahead=None, subgoal_offset=0.0, lane_hold=LANE_HOLD_RANGE):',
    'signature')

sub('src/cash_rl/cash_rl/raycast_env.py',
    '        self.pedestrians_yield = pedestrians_yield\n'
    '        self.lane_penalty = lane_penalty\n',
    '        self.pedestrians_yield = pedestrians_yield\n'
    '        self.yield_prob = float(yield_prob)\n'
    '        self.cross_prob = float(cross_prob)\n'
    '        self.lane_penalty = lane_penalty\n',
    'store')

sub('src/cash_rl/cash_rl/raycast_env.py',
    '        self._pedestrian_velocities = np.zeros((0, 2))\n'
    '        self._step_count = 0\n',
    '        self._pedestrian_velocities = np.zeros((0, 2))\n'
    '        self._pedestrian_yields = np.zeros(0, dtype=bool)\n'
    '        self._step_count = 0\n',
    'init yields')

ASSIGN = '''
        self._assign_behaviors()

    def _assign_behaviors(self):
        """Mark who steps aside, and send crossers across the corridor.

        Neither draw runs unless its probability is set, so yield_prob and
        cross_prob at 0 leave a seeded episode identical to the old discs.
        A crosser keeps their line (the robot waits); everyone else yields
        with yield_prob. pedestrians_yield still means every person yields.
        """
        n = len(self.pedestrians)
        yields = np.ones(n, dtype=bool) if self.pedestrians_yield else np.zeros(n, dtype=bool)
        if n and not self.pedestrians_yield and (self.yield_prob > 0 or self.cross_prob > 0):
            for i in range(n):
                draw = float(self.np_random.random())
                if draw < self.cross_prob:
                    axis = self._walking_axis(self.pedestrians[i])
                    cross = np.array([-axis[1], axis[0]])
                    sign = 1.0 if self.np_random.random() < 0.5 else -1.0
                    speed = float(self.np_random.uniform(*CROSS_SPEED_RANGE))
                    self._pedestrian_velocities[i] = sign * speed * cross
                elif draw < self.cross_prob + (1.0 - self.cross_prob) * self.yield_prob:
                    yields[i] = True
        self._pedestrian_yields = yields

'''

sub('src/cash_rl/cash_rl/raycast_env.py',
    '        self._pedestrian_velocities = np.array(velocities).reshape(-1, 2)\n'
    '    def route(self, max_length=None):\n',
    '        self._pedestrian_velocities = np.array(velocities).reshape(-1, 2)\n'
    + ASSIGN
    + '    def route(self, max_length=None):\n',
    'assign')

sub('src/cash_rl/cash_rl/raycast_env.py',
    '        if self.pedestrians_yield:\n'
    '            sidesteps, yielding, in_path = self._sidesteps(dt)\n'
    '            self.pedestrians += sidesteps\n'
    '            stop = in_path & (self.pedestrian_gaps() < PEDESTRIAN_STOP_DISTANCE)\n'
    '            speed_factor = np.where(stop, 0.0, np.where(yielding, PEDESTRIAN_SLOW_FACTOR, 1.0))\n',
    '        if len(self._pedestrian_yields) and np.any(self._pedestrian_yields):\n'
    '            sidesteps, yielding, in_path = self._sidesteps(dt)\n'
    '            mask = self._pedestrian_yields\n'
    '            sidesteps = sidesteps * mask[:, None]\n'
    '            yielding = yielding & mask\n'
    '            in_path = in_path & mask\n'
    '            self.pedestrians += sidesteps\n'
    '            stop = in_path & (self.pedestrian_gaps() < PEDESTRIAN_STOP_DISTANCE)\n'
    '            speed_factor = np.where(stop, 0.0, np.where(yielding, PEDESTRIAN_SLOW_FACTOR, 1.0))\n',
    'move')

sub('src/cash_rl/cash_rl/decision.py',
    'PASS_SEPARATION = 0.8\n',
    'PASS_SEPARATION = 0.8\n'
    'PASS_CLEARANCE = 0.545\n',
    'clearance')

OLD_LAYER = '''    def __init__(self, env, decider=None, prob=1.0):
        super().__init__(env)
        if not env.unwrapped.subgoal_lookahead:
            raise ValueError('the decision layer needs subgoal_lookahead')
        self.decider = decider or PedestrianLaneDecider()
        self.prob = prob
        self.active = True
        self._rng = np.random.default_rng()

    def _decide(self, info):
        base = self.env.unwrapped
        base.lane_offset = float(self.decider(self.env))
        obs = base.observation()  # same scans, subgoal in the new lane
        info['decision_offset'] = base.lane_offset
        info['subgoal'] = base.subgoal.copy()
        if base.lane is not None:
            info['lane_offset'] = base.lane[2]
        return obs, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.active = self.prob >= 1 or self._rng.random() < self.prob
        if not self.active:
            return self.env.reset(seed=seed, options=options)
        self.decider.reset()
        options = dict(options or {}, lane_offset=0.0)
        _, info = self.env.reset(seed=seed, options=options)
        return self._decide(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.active and not (terminated or truncated):
            obs, info = self._decide(info)
        return obs, reward, terminated, truncated, info
'''

NEW_LAYER = '''    def __init__(self, env, decider=None, prob=1.0, wait=False):
        super().__init__(env)
        if not env.unwrapped.subgoal_lookahead:
            raise ValueError('the decision layer needs subgoal_lookahead')
        self.decider = decider or PedestrianLaneDecider()
        self.prob = prob
        self.wait = wait
        self.active = True
        self._waiting = False
        self._rng = np.random.default_rng()

    def _pass_blocked(self):
        """True when someone ahead still overlaps the lane the robot would drive.

        With no lane (the last metres before the goal) it is someone just in
        front of the footprint. Checked again every step, so the hold ends once
        they have passed or a clear lane appears.
        """
        base = self.env.unwrapped
        if len(base.pedestrians) == 0:
            return False
        if base.near_lane is not None:
            point, direction, applied = base.near_lane
            normal = np.array([-direction[1], direction[0]])
            center = point - applied * normal
            rel = base.pedestrians - base.pose[:2]
            ahead = rel @ direction
            gap = np.abs((base.pedestrians - center) @ normal - applied)
            blocked = (ahead > -0.3) & (ahead < MAX_AHEAD) & (gap < PASS_CLEARANCE)
            return bool(np.any(blocked))
        gaps = base.pedestrian_gaps()
        forward = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
        ahead = (base.pedestrians - base.pose[:2]) @ forward
        return bool(np.any((ahead > 0.0) & (ahead < 2.5) & (gaps < 0.35)))

    def _decide(self, info):
        base = self.env.unwrapped
        base.lane_offset = float(self.decider(self.env))
        obs = base.observation()  # same scans, subgoal in the new lane
        self._waiting = bool(self.wait and self._pass_blocked())
        info['decision_offset'] = base.lane_offset
        info['decision_wait'] = self._waiting
        info['subgoal'] = base.subgoal.copy()
        if base.lane is not None:
            info['lane_offset'] = base.lane[2]
        return obs, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.active = self.prob >= 1 or self._rng.random() < self.prob
        self._waiting = False
        if not self.active:
            return self.env.reset(seed=seed, options=options)
        self.decider.reset()
        options = dict(options or {}, lane_offset=0.0)
        _, info = self.env.reset(seed=seed, options=options)
        return self._decide(info)

    def step(self, action):
        waited = self.active and self._waiting
        if waited:
            action = np.zeros(2, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['waited'] = waited
        if self.active and not (terminated or truncated):
            obs, info = self._decide(info)
            info['waited'] = waited
        return obs, reward, terminated, truncated, info
'''

sub('src/cash_rl/cash_rl/decision.py', OLD_LAYER, NEW_LAYER, 'decision')

sub('src/cash_rl/cash_rl/tracks.py',
    '    def __init__(self, env, decider=None, prob=1.0):\n'
    '        super().__init__(env, decider=decider, prob=prob)\n',
    '    def __init__(self, env, decider=None, prob=1.0, wait=False):\n'
    '        super().__init__(env, decider=decider, prob=prob, wait=wait)\n',
    'tracks')

sub('src/cash_rl/cash_rl/evaluate.py',
    "    parser.add_argument('--yielding-pedestrians', action='store_true',\n"
    "                        help='pedestrians step around the robot instead of walking into it')\n",
    "    parser.add_argument('--yielding-pedestrians', action='store_true',\n"
    "                        help='pedestrians step around the robot instead of walking into it')\n"
    "    parser.add_argument('--yield-prob', type=float, default=0.0,\n"
    "                        help='chance each pedestrian steps aside. 0 keeps the old discs')\n"
    "    parser.add_argument('--cross-prob', type=float, default=0.0,\n"
    "                        help='chance each pedestrian crosses the corridor at 1.0-1.4 m/s')\n",
    'eval flags')

sub('src/cash_rl/cash_rl/evaluate.py',
    "                             'lookahead)')\n"
    "    parser.add_argument('--scan-tracks', action='store_true',\n",
    "                             'lookahead)')\n"
    "    parser.add_argument('--wait', action='store_true',\n"
    "                        help='with --decision-layer, hold still while the chosen lane still '\n"
    "                             'leaves a person inside the robot')\n"
    "    parser.add_argument('--scan-tracks', action='store_true',\n",
    'eval wait')

sub('src/cash_rl/cash_rl/evaluate.py',
    '    if args.scan_tracks and not args.decision_layer:\n',
    '    if args.wait and not args.decision_layer:\n'
    "        parser.error('--wait is part of --decision-layer')\n"
    '    if args.scan_tracks and not args.decision_layer:\n',
    'eval check')

sub('src/cash_rl/cash_rl/evaluate.py',
    '        env = HallwayRaycastEnv(n_pedestrians=args.pedestrians,\n'
    '                                pedestrians_yield=args.yielding_pedestrians,\n'
    '                                encounter_prob=args.encounter_prob,\n',
    '        env = HallwayRaycastEnv(n_pedestrians=args.pedestrians,\n'
    '                                pedestrians_yield=args.yielding_pedestrians,\n'
    '                                yield_prob=args.yield_prob,\n'
    '                                cross_prob=args.cross_prob,\n'
    '                                encounter_prob=args.encounter_prob,\n',
    'eval env')

sub('src/cash_rl/cash_rl/evaluate.py',
    '            env = ScanDecisionLayer(env)\n'
    '        else:\n'
    '            from cash_rl.decision import DecisionLayer\n'
    '            env = DecisionLayer(env)\n',
    '            env = ScanDecisionLayer(env, wait=args.wait)\n'
    '        else:\n'
    '            from cash_rl.decision import DecisionLayer\n'
    '            env = DecisionLayer(env, wait=args.wait)\n',
    'eval wrap')

sub('src/cash_rl/cash_rl/evaluate.py',
    "        + (' (yielding)' if args.pedestrians and args.yielding_pedestrians else '') \\\n",
    "        + (' (yielding)' if args.pedestrians and args.yielding_pedestrians else '') \\\n"
    "        + (f', yield {args.yield_prob:g}' if args.pedestrians and args.yield_prob else '') \\\n"
    "        + (f', cross {args.cross_prob:g}' if args.pedestrians and args.cross_prob else '') \\\n",
    'eval mode people')

sub('src/cash_rl/cash_rl/evaluate.py',
    "           else ', decision layer' if args.decision_layer else '') \\\n",
    "           else ', decision layer' if args.decision_layer else '') \\\n"
    "        + (', wait' if args.wait else '') \\\n",
    'eval mode wait')

sub('src/cash_rl/cash_rl/dagger.py',
    'def _init_worker(env_kwargs, decision_layer=False, teacher=\'avoider\'):\n',
    'def _init_worker(env_kwargs, decision_layer=False, teacher=\'avoider\', wait=False):\n',
    'worker sig')

sub('src/cash_rl/cash_rl/dagger.py',
    '        env = DecisionLayer(env)\n'
    '    _worker_env = env\n',
    '        env = DecisionLayer(env, wait=wait)\n'
    '    _worker_env = env\n',
    'worker wrap')

sub('src/cash_rl/cash_rl/dagger.py',
    "                             \"the robot's route, walking toward it\")\n",
    "                             \"the robot's route, walking toward it\")\n"
    "    parser.add_argument('--yield-prob', type=float, default=0.0,\n"
    "                        help='chance each pedestrian steps aside (0 keeps the old discs)')\n"
    "    parser.add_argument('--cross-prob', type=float, default=0.0,\n"
    "                        help='chance each pedestrian crosses the corridor at 1.0-1.4 m/s')\n",
    'dagger flags')

sub('src/cash_rl/cash_rl/dagger.py',
    "                             'the per-round eval (needs --subgoal-lookahead)')\n",
    "                             'the per-round eval (needs --subgoal-lookahead)')\n"
    "    parser.add_argument('--wait', action='store_true',\n"
    "                        help='with --decision-layer, hold still while the chosen lane still '\n"
    "                             'leaves a person inside the robot')\n",
    'dagger wait')

sub('src/cash_rl/cash_rl/dagger.py',
    "        parser.error('--decision-layer needs --subgoal-lookahead')\n",
    "        parser.error('--decision-layer needs --subgoal-lookahead')\n"
    '    if args.wait and not args.decision_layer:\n'
    "        parser.error('--wait is part of --decision-layer')\n",
    'dagger check')

sub('src/cash_rl/cash_rl/dagger.py',
    "    env_kwargs = {'n_pedestrians': args.pedestrians, 'encounter_prob': args.encounter_prob,\n"
    "                  'subgoal_lookahead': args.subgoal_lookahead,\n",
    "    env_kwargs = {'n_pedestrians': args.pedestrians, 'encounter_prob': args.encounter_prob,\n"
    "                  'yield_prob': args.yield_prob, 'cross_prob': args.cross_prob,\n"
    "                  'subgoal_lookahead': args.subgoal_lookahead,\n",
    'dagger kwargs')

sub('src/cash_rl/cash_rl/dagger.py',
    '    eval_env = HallwayRaycastEnv(n_pedestrians=args.pedestrians, encounter_prob=1.0,\n'
    '                                 subgoal_lookahead=args.subgoal_lookahead,\n',
    '    eval_env = HallwayRaycastEnv(n_pedestrians=args.pedestrians, encounter_prob=1.0,\n'
    '                                 yield_prob=args.yield_prob, cross_prob=args.cross_prob,\n'
    '                                 subgoal_lookahead=args.subgoal_lookahead,\n',
    'dagger eval env')

sub('src/cash_rl/cash_rl/dagger.py',
    '        eval_env = DecisionLayer(eval_env)\n',
    '        eval_env = DecisionLayer(eval_env, wait=args.wait)\n',
    'dagger eval wrap')

sub('src/cash_rl/cash_rl/dagger.py',
    "          + (', decision layer' if args.decision_layer else '')\n"
    "          + f' -> {run_dir}', flush=True)\n",
    "          + (', decision layer' if args.decision_layer else '')\n"
    "          + (', wait' if args.wait else '')\n"
    "          + (f', yield {args.yield_prob:g}' if args.yield_prob else '')\n"
    "          + (f', cross {args.cross_prob:g}' if args.cross_prob else '')\n"
    "          + f' -> {run_dir}', flush=True)\n",
    'dagger banner')

sub('src/cash_rl/cash_rl/dagger.py',
    '                      initargs=(env_kwargs, args.decision_layer, args.teacher)) as pool:\n',
    '                      initargs=(env_kwargs, args.decision_layer, args.teacher,\n'
    '                                args.wait)) as pool:\n',
    'dagger pool')

print('PATCH_OK')
