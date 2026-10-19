package br.com.guialves.rflr.gymnasium4j.stableretro;

import br.com.guialves.rflr.gymnasium4j.Gym;

public class StableRetro extends Gym {

    public static StableRetroEnvBuilder builder() {
        return new StableRetroEnvBuilder();
    }

    public static class StableRetroEnvBuilder extends EnvBuilder {

        private List<String> envsNames;

        protected StableRetroEnvBuilder() {
            importLib("stable_retro as retro");
        }

        public StableRetroEnvBuilder envsNames(@NonNull List<String> envsNames) {
            this.envsNames = envsNames;
            return this;
        }

        protected String generateMakeCall() {
            if (params.isEmpty()) {
                return "gym.make('%s', render_mode='rgb_array')".formatted(envName);
            }

            return "gym.make('%s', render_mode='rgb_array', %s)".formatted(
                    envName, params.toPyKwargs());
        }
    }
}
