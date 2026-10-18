import stable_retro as retro

def main():

    megamans_games = ["MegaMan-Nes-v0", "MegaMan2-Nes-v0", "MegaManX-Snes"]

    for game in megamans_games:
        env = retro.make(game=game, inttype=retro.data.Integrations.ALL)
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            env.render()
            if term or trunc:
                break

        env.close()

if __name__ == "__main__":
    main()
