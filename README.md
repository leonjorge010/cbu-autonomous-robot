# C.A.$.H Robot Workspace

ROS 2 Jazzy workspace for the C.A.$.H autonomous hallway escort robot.
Runs on Ubuntu 24.04 in WSL2 with Gazebo Harmonic.

## Start working (from computer power-on)

1. **Open Ubuntu**
   - Windows Terminal → click the ⌄ arrow → **Ubuntu 24.04**
   - Or in PowerShell: `wsl -d Ubuntu-24.04`

2. **Go to the workspace and get the latest code**
```bash
   cd ~/ros2_ws
   git pull
```

3. **Install any new dependencies** (only needed if packages were added)
```bash
   rosdep install --from-paths src -y --ignore-src
```

4. **Build and source**
```bash
   colcon build --symlink-install
   source install/setup.bash
```

5. **Open the editor** (optional)
```bash
   code .
```

6. **Extra terminals**: press `Alt+Shift+D` for a split pane, then in the new pane:
```bash
   cd ~/ros2_ws && source install/setup.bash
```

## Save and push work

```bash
git status
git add .
git commit -m "Describe what changed"
git push
```

## Quick checks

| Check | Command | Expected |
|---|---|---|
| GPU rendering | `glxinfo -B \| grep -i renderer` | `D3D12 (NVIDIA GeForce RTX 4060 ...)` |
| Gazebo | `gz sim shapes.sdf` | Window with shapes opens |
| ROS comms | `ros2 run demo_nodes_cpp talker` + `listener` in a 2nd pane | Listener prints `I heard: ...` |
| GitHub login | `gh auth status` | `Logged in to github.com account leonjorge010` |

## Troubleshooting

- **Renderer says `llvmpipe` (slow, CPU rendering)**: make sure `~/.bashrc` contains
  `export GALLIUM_DRIVER=d3d12` and `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`, then `source ~/.bashrc`.
- **`git push` / `gh` asks to log in**: the access token may have expired. Generate a new classic token
  (scopes: repo, workflow, read:org, admin:public_key) and run `gh auth login` → Paste an authentication token.
- **Package not found after building**: you forgot `source install/setup.bash` in that terminal.
- **Slow builds**: make sure you're in `~/ros2_ws`, not under `/mnt/c/...`.
