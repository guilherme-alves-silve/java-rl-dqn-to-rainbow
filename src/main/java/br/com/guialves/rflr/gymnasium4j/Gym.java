package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.wrappers.IWrapper;
import lombok.SneakyThrows;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

import static java.util.Objects.requireNonNull;

@Slf4j
public class Gym {

    @SneakyThrows
    public static Env make(String name,
                           NDManager ndManager) {
        return builder()
                .envName(name)
                .ndManager(ndManager)
                .build();
    }

    public static EnvBuilder builder() {
        return new EnvBuilder();
    }

    public static PyMap builderMap() {
        return new PyMap();
    }

    public static class EnvBuilder {

        private static final PyMap DEFAULT_MAP = new PyMap();
        private final String varEnvCode;
        private final List<IWrapper> wrappers;
        private String envName;
        private PyMap params;
        private NDManager ndManager;

        private EnvBuilder() {
            this.varEnvCode = UUID.randomUUID().toString().replace("-", "");
            this.wrappers = new ArrayList<>();
            this.params = DEFAULT_MAP;
        }

        public EnvBuilder envName(String envName) {
            this.envName = envName;
            return this;
        }

        public EnvBuilder ndManager(NDManager ndManager) {
            this.ndManager = ndManager;
            return this;
        }

        public EnvBuilder params(PyMap params) {
            this.params = params;
            return this;
        }

        public EnvBuilder add(IWrapper wrapper) {
            wrappers.add(requireNonNull(wrapper, "wrapper cannot be null!"));
            return this;
        }

        public EnvBuilder add(IWrapper wrapper, IWrapper... wrappersArray) {
            add(wrapper);
            Arrays.stream(requireNonNull(wrappersArray, "wrappersArray cannot be null!"))
                    .forEach(this::add);
            return this;
        }

        String generatePyEnvScript() {
            if (wrappers.isEmpty()) {
                return """
                import gymnasium as gym
                env_%s = gym.make('%s', render_mode='rgb_array')
                """.formatted(varEnvCode, envName);
            }

            var importPy = generateImportPy();
            var wrappedEnvPy = generateWrappedEnvPy();

            return """
            import gymnasium as gym
            %s
            env_%s = gym.make('%s', render_mode='rgb_array')
            %s
            """.formatted(importPy, varEnvCode, envName, wrappedEnvPy);
        }

        private String generateImportPy() {
            var builder = new StringBuilder("from gymnasium.wrappers import ");
            for (int i = 0; i < wrappers.size() - 1; ++i) {
                builder.append(wrappers.get(i).getClass().getSimpleName())
                        .append(", ");
            }

            return builder.append(wrappers.getLast().getClass().getSimpleName())
                    .toString();
        }

        private String generateWrappedEnvPy() {
            var varName = "env_" + varEnvCode;
            var builder = new StringBuilder();
            wrappers.forEach(wrapper -> builder.append(varName)
                    .append(" = ")
                    .append(wrapper.pyToStr(varName))
                    .append(System.lineSeparator()));
            return builder.toString();
        }

        public Env build() {
            return new Env(requireNonNull(varEnvCode, "varEnvCode cannot be null!"),
                    requireNonNull(envName, "envId cannot be null!"),
                    generatePyEnvScript(),
                    requireNonNull(ndManager, "cannot be null!"));
        }
    }

    /**
     * Builder for Python keyword arguments (kwargs) used in gym.make()
     * Example: PyMap with domain_randomize=True, continuous=False
     * becomes: domain_randomize=True, continuous=False
     */
    public static class PyMap {

        private final Map<String, String> params;

        private PyMap() {
            this.params = new LinkedHashMap<>();
        }

        /**
         * Add a parameter with its value
         * @param key parameter name
         * @param value parameter value (will be used as-is in Python)
         * @return this builder for chaining
         */
        public PyMap put(String key, String value) {
            params.put(key, value);
            return this;
        }

        /**
         * Add a boolean parameter
         * @param key parameter name
         * @param value boolean value (converted to Python True/False)
         * @return this builder for chaining
         */
        public PyMap put(String key, boolean value) {
            params.put(key, value ? "True" : "False");
            return this;
        }

        /**
         * Add an integer parameter
         * @param key parameter name
         * @param value integer value
         * @return this builder for chaining
         */
        public PyMap put(String key, int value) {
            params.put(key, String.valueOf(value));
            return this;
        }

        /**
         * Add a double parameter
         * @param key parameter name
         * @param value double value
         * @return this builder for chaining
         */
        public PyMap put(String key, double value) {
            params.put(key, String.valueOf(value));
            return this;
        }

        /**
         * Add a string parameter (will be quoted in Python)
         * @param key parameter name
         * @param value string value
         * @return this builder for chaining
         */
        public PyMap putStr(String key, String value) {
            params.put(key, "'" + value + "'");
            return this;
        }

        /**
         * Check if map is empty
         * @return true if no parameters have been added
         */
        public boolean isEmpty() {
            return params.isEmpty();
        }

        /**
         * Get number of parameters
         * @return parameter count
         */
        public int size() {
            return params.size();
        }

        /**
         * Convert to Python kwargs string format
         * Example: "domain_randomize=True, continuous=False"
         * @return Python kwargs string
         */
        public String toPyKwargs() {
            return params.entrySet()
                    .stream()
                    .map(entry -> entry.getKey() + "=" + entry.getValue())
                    .collect(Collectors.joining(", "));
        }

        /**
         * Convert to Python dict string format
         * Example: "{'domain_randomize': True, 'continuous': False}"
         * @return Python dict string
         */
        public String toPyDict() {
            return params.entrySet()
                    .stream()
                    .map(entry -> "'" + entry.getKey() + "': " + entry.getValue())
                    .collect(Collectors.joining(", ", "{", "}"));
        }

        @Override
        public String toString() {
            return toPyKwargs();
        }
    }
}
