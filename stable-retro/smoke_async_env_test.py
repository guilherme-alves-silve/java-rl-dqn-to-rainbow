import stable_retro as retro
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


def make_env(game: str):
    def _init():
        return retro.make(game=game, inttype=retro.data.Integrations.ALL)
    return _init


def main(n_envs=4):
    megamans_games = ["MegaMan-Nes-v0", "MegaMan2-Nes-v0"]
    async_envs = AsyncVectorEnv([make_env(game) for game in megamans_games for _ in range(n_envs)], context="spawn")
    async_envs.reset()
    for _ in range(50):
        actions = async_envs.action_space.sample()
        async_envs.step(actions)
    async_envs.close()


if __name__ == "__main__":
    main()
