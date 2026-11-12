"""The PPO actor stays put during a value-head warmup."""
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.train import set_actor_frozen
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def test_value_warmup_freezes_only_the_actor():
    env = DummyVecEnv([lambda: HallwayRaycastEnv(range_noise=False)])
    model = PPO('MlpPolicy', env, n_steps=32, batch_size=32, n_epochs=1, verbose=0)
    set_actor_frozen(model.policy, True)
    assert model.policy.log_std.requires_grad is False
    assert all(not p.requires_grad for p in model.policy.action_net.parameters())
    assert all(not p.requires_grad for p in model.policy.mlp_extractor.policy_net.parameters())
    assert any(p.requires_grad for p in model.policy.value_net.parameters())
    assert any(p.requires_grad for p in model.policy.mlp_extractor.value_net.parameters())
    set_actor_frozen(model.policy, False)
    assert model.policy.log_std.requires_grad is True
    assert all(p.requires_grad for p in model.policy.action_net.parameters())
    env.close()
