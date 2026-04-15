from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='easycarla.envs:CarlaEnv',
)