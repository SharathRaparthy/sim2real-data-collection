from gym.envs.registration import register

register(
    id='Pusher3DOFDefault-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/default.json'}
)

register(
    id='Pusher3DOFRandomized-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/random.json'}
)