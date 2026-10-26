package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.gymnasium4j.wrappers.IWrapper;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

import static java.util.stream.Collectors.joining;

/**
 * Creates the Gymnasium vector environment (Sync or Async) from Java.
 */
@Slf4j
public class VectorGym extends Gym {

    @SneakyThrows
    public static IEnv make(@NonNull String name) {
        log.warn("It's recommended to use makeSync or makeAsync instead!");
        return make(name, false, 1);
    }

    @SneakyThrows
    public static IEnv makeVec(@NonNull String name, boolean async, int numberOfEnvs) {
        return make(name, async, numberOfEnvs);
    }

    @SneakyThrows
    private static IEnv make(@NonNull String name, boolean async, int numberOfEnvs) {
        return builder()
                .envName(name)
                .numberOfEnvs(numberOfEnvs)
                .async(async)
                .build();
    }

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
            var correctionMultiprocessingWhenAsync = generateCorrectionMultiprocessingWhenAsync();
            var wrappedEnvParams = generateWrappedEnvPy();
            var makeCall = generateMakeCall();
            var importLibsPy = generateImportLibsPy();
            var functionDefinitionsCode = generateFunctionDefinitions();
            if (wrappers.isEmpty()) {
                return """
                %s%s
                %s
                %s
                env_%s = %s
                """.formatted(correctionMultiprocessingWhenAsync, importLibsPy, functionDefinitionsCode,
                        wrappedEnvParams, varEnvCode, makeCall);
            }

            var importPy = generateImportFromPy();

            return """
            %s%s
            %s
            %s
            %s
            env_%s = %s
            """.formatted(correctionMultiprocessingWhenAsync, importLibsPy, importPy, functionDefinitionsCode,
                    wrappedEnvParams, varEnvCode, makeCall);
        }

        /**
         * <h4>Why {@code AsyncVectorEnv} fails with "Unrecognized option: -c"</h4>
         * <p>
         * {@code AsyncVectorEnv} relies on {@code multiprocessing} to create its workers. With
         * the {@code spawn} and {@code forkserver} start methods, each worker is launched by
         * running {@code [sys.executable, "-c", "from multiprocessing.spawn import
         * spawn_main; ...", "--multiprocessing-fork"]}.
         * <p>
         * When Python is embedded in the JVM through JavaCPP, {@code sys.executable} points to
         * the {@code java} binary rather than a Python interpreter. Each worker therefore
         * becomes {@code java -c ...}, and the JVM rejects the {@code -c} option and exits.
         * One error message is printed per worker.
         *
         * <h4>Why {@code SyncVectorEnv} works</h4>
         * <p>
         * {@code SyncVectorEnv} does not create subprocesses. All environments run in the same
         * process (and the same JVM), so {@code sys.executable} is never used.
         *
         * <h4>Fix for {@code AsyncVectorEnv}</h4>
         * <pre>{@code
         * import multiprocessing, os, sys
         * py = os.path.join(sys.exec_prefix, "python.exe" if os.name == "nt" else "bin/python")
         * multiprocessing.set_executable(py)   // before creating the AsyncVectorEnv
         * gym.vector.AsyncVectorEnv(env_fns, context="spawn")
         * }</pre>
         *
         * <h4>Constraints</h4>
         * <ul>
         *   <li>Workers are plain Python processes with <b>no JVM</b>. The {@code env_fns}
         *       must be picklable and importable in the child process, and must not depend on
         *       JavaCPP objects or Java callbacks.</li>
         *   <li>{@code context} is an {@code AsyncVectorEnv}-only parameter.
         *       {@code SyncVectorEnv} does not accept it.</li>
         *   <li>Avoid {@code fork}: forking a process that hosts a JVM with live threads tends
         *       to cause deadlocks.</li>
         *   <li>Verify that the interpreter path exists. The value of {@code sys.exec_prefix}
         *       depends on how JavaCPP configured the embedded Python.</li>
         * </ul>
         *
         * @implNote If the environment depends on the Java side, use {@code SyncVectorEnv}, or
         *           parallelize on the Java side with several {@code Env} instances.
         *
         * @see <a href="https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_executable">multiprocessing.set_executable (Python docs)</a>
         * @see <a href="https://stackoverflow.com/questions/15636266/embedded-python-multiprocessing-not-working">Stack Overflow: Embedded Python multiprocessing not working</a>
         * @see <a href="https://github.com/pybind/pybind11/pull/5238">pybind11 #5238: multiprocessing spawn test in an embedded interpreter</a>
         * @see <a href="https://pyoxidizer.readthedocs.io/en/stable/pyoxidizer_packaging_multiprocessing.html">PyOxidizer: multiprocessing and sys.executable sensitivity</a>
         */
        private String generateCorrectionMultiprocessingWhenAsync() {
            if (!async) {
                return "";
            }

            return """
            import multiprocessing, os, sys
            if os.name == "nt":
                py = os.path.join(sys.exec_prefix, "python.exe")
            else:
                py = os.path.join(sys.exec_prefix, "bin", "python")
            multiprocessing.set_executable(py)
            
            """;
        }

        @Override
        public Env build() {
            if (async) {
                params.put("context", "'spawn'");
            }

            return new Env(varEnvCode, envName, generatePyEnvScript());
        }
    }
}
