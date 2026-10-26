package br.com.guialves.rflr.gymnasium4j;

public class StableRetroGym extends VectorGym {

    public static StableRetroEnvBuilder builder() {
        return new StableRetroEnvBuilder();
    }

    public static class StableRetroEnvBuilder extends VectorEnvBuilder {

        protected StableRetroEnvBuilder() {
            super.importLib("stable_retro as retro");
            // remove the definition of make_env. Not necessary.
            super.functionDefinitions.removeLast();
            super.functionDefinitions.add("""
            def make_retro_env(idx: int, env_wrappers: tuple[str, dict]):
                def _init_retro():
                    env_name, wrappers = env_wrappers
                    env = retro.make(game=env_name, render_mode='rgb_array', inttype=retro.data.Integrations.ALL)
                    for wrapper, wrapper_params in wrappers:
                        params = dict(wrapper_params)
                        if 'video_folder' in wrapper_params:
                            params['video_folder'] = f"{params['video_folder']}/{env_name}{idx}"
                        env = wrapper(env, **params)
                    return env
                return _init_retro
            """);
            super.setFunctionToCall("make_retro_env");
        }
    }
}
