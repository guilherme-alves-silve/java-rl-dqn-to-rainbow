package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.gymnasium4j.wrappers.IWrapper;
import lombok.NonNull;

import java.util.List;

import static java.util.stream.Collectors.joining;

public class VectorGym extends Gym {

    public static VectorEnvBuilder builder() {
        return new VectorEnvBuilder();
    }

    public static non-sealed class VectorEnvBuilder extends EnvBuilder {

        private boolean async;
        private int numberOfEnvs;
        private List<String> envsNames;
        private String functionToCall;

        protected VectorEnvBuilder() {
            this.numberOfEnvs = 1;
            this.functionToCall = "make_env";
        }

        @Override
        public VectorEnvBuilder envName(@NonNull String envName) {
            this.envName = envName;
            return envsNames(List.of(envName));
        }

        public VectorEnvBuilder numberOfEnvs(int numberOfEnvs) {
            if (numberOfEnvs <= 0) throw new IllegalArgumentException("numberOfEnvs must be greater than 0");
            this.numberOfEnvs = numberOfEnvs;
            return this;
        }

        public VectorEnvBuilder async(boolean async) {
            this.async = async;
            return this;
        }

        protected final void setFunctionToCall(@NonNull String functionToCall) {
            this.functionToCall = functionToCall;
        }

        public VectorEnvBuilder envsNames(@NonNull List<String> envsNames) {
            this.envsNames = envsNames;
            return this;
        }

        protected String generateWrappedEnvPy() {
            if (wrappers.isEmpty()) {
                return "env_params = [ ]";
            }

            var builder = new StringBuilder()
                    .append("env_params = [ ");
            for (int i = 0; i < wrappers.size() - 1; ++i) {
                appendWrapper(builder, wrappers.get(i)).append(", ");
            }
            appendWrapper(builder, wrappers.getLast());
            return builder.append(" ]").toString();
        }

        protected final StringBuilder appendWrapper(StringBuilder builder, IWrapper wrapper) {
            return builder.append("(")
                    .append(wrapper.getClass().getSimpleName())
                    .append(" , ")
                    .append(wrapper.pyKwargs())
                    .append(")");
        }

        protected String generateMakeCall() {
            var tupleEnvs = envsNames.stream()
                    .map("('%s', env_params)"::formatted)
                    .collect(joining(", "));
            var envsBuilding = "[%s(idx, env_wrappers) for env_wrappers in [%s] for idx in range(%d)]".formatted(
                    functionToCall, tupleEnvs, numberOfEnvs);
            var vectorType = async? "AsyncVectorEnv" : "SyncVectorEnv";
            if (params.isEmpty()) {
                return "gym.vector.%s(%s)".formatted(vectorType, envsBuilding);
            }

            return "gym.vector.%s(%s, %s)".formatted(
                    vectorType, envsBuilding, params.toPyKwargs());
        }

        protected String generatePyEnvScript() {
            var wrappedEnvParams = generateWrappedEnvPy();
            var makeCall = generateMakeCall();
            var importLibsPy = generateImportLibsPy();
            var functionDefinitionsCode = generateFunctionDefinitions();
            if (wrappers.isEmpty()) {
                return """
                %s
                %s
                %s
                env_%s = %s
                """.formatted(importLibsPy, functionDefinitionsCode, wrappedEnvParams, varEnvCode, makeCall);
            }

            var importPy = generateImportFromPy();

            return """
            %s
            %s
            %s
            %s
            env_%s = %s
            """.formatted(importLibsPy, importPy, functionDefinitionsCode, wrappedEnvParams, varEnvCode, makeCall);
        }
    }
}
