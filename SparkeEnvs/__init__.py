from gymnasium.envs.registration import register

register(
    id="rl_bullet_test_envs/TestEnv-v0",
    entry_point="rl_bullet_test_envs.envs:TestEnv",
    max_episode_steps=1000,
)