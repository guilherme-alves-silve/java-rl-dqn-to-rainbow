import gymnasium as gym
from gymnasium.wrappers import DelayObservation, RecordVideo
from gymnasium.vector import AsyncVectorEnv


def make_env(idx: int, env_wrappers: tuple[str, dict]):
    def _init():
        env_name, wrappers = env_wrappers
        env = gym.make(game=env_name, render_mode='rgb_array')
        video_folder_backup = ''
        for wrapper, wrapper_params in wrappers:
            if 'video_folder' in wrapper_params:
                video_folder_backup = wrapper_params['video_folder']
                wrapper_params['video_folder'] = f"{wrapper_params['video_folder']}/{env_name}{idx}"
            env = wrapper(env, **wrapper_params)
            if 'video_folder' in wrapper_params:
                wrapper_params['video_folder'] = video_folder_backup
        return env
    return _init


def main(n_envs=4):
    env_params = [(DelayObservation, {'delay': 1}),
                  (RecordVideo, {'video_folder': '/tmp/video', 'video_length': 500, 'fps': 30})]
    env = AsyncVectorEnv([make_env(idx, env_wrappers) for env_wrappers in
                           [('CartPole-v0', env_params)]
                           for idx in range(n_envs)], context="spawn")
    env.reset()
    for _ in range(50):
        actions = env.action_space.sample()
        env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
