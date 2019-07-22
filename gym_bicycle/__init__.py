from gym.envs.registration import register

register(
    id='bicycle-v0',
    entry_point='gym.envs.classic_control:BicycleEnv',
)