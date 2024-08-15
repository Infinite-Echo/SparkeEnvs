from gymnasium.envs.registration import register

register(
    id="SparkeEnvs/SparkeEnv-v0",
    entry_point="SparkeEnvs.envs:SparkeEnv",
)