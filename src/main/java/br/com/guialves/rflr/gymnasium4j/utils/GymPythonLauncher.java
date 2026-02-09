package br.com.guialves.rflr.gymnasium4j.utils;

import ai.djl.util.JsonUtils;
import com.google.gson.Gson;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Slf4j
public class GymPythonLauncher implements AutoCloseable {

    private static final int WAIT_START = 2000;
    private static final Gson GSON = JsonUtils.GSON_COMPACT;
    private final String projectRoot;
    private final String pythonPath;
    private final String scriptPath;
    private final int port;
    private final int timeout;
    private final boolean debug;
    private final String envName;
    private final Map<String, Object> envParams;
    private Process serverProcess;

    public GymPythonLauncher(String scriptPath,
                             int port,
                             int timeout,
                             boolean debug,
                             String envName,
                             Map<String, Object> envParams) {
        this.scriptPath = scriptPath;
        this.port = port;
        this.timeout = timeout;
        this.debug = debug;
        this.envName = envName;
        this.envParams = envParams;
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        this.projectRoot = Paths.get(System.getProperty("user.dir"), "gymnasium").toString();
        this.pythonPath = isWindows
                ? Paths.get(projectRoot, ".venv", "Scripts", "python.exe").toString()
                : Paths.get(projectRoot, ".venv", "bin", "python").toString();
    }

    public void setupEnvironment() {
        log.info("Checking Python environment setup...");

        String versionOutput = runCommand(List.of("python", "--version"));
        log.info("System Python: {}", versionOutput.trim());

        if (!versionOutput.contains("3.11") && !versionOutput.contains("3.12") && !versionOutput.contains("3.13")) {
            log.warn("Warning: System Python version might be lower than 3.11. Current: {}", versionOutput);
        }

        var out1 = runCommand(List.of("pip", "install", "uv"));
        log.info(out1);

        var venvDir = new File(projectRoot, ".venv");
        if (!venvDir.exists()) {
            log.info("Creating virtual environment in {} using uv...", projectRoot);
            var out2 = runCommand(List.of("uv", "venv", "--python", "3.12"), new File(projectRoot));
            log.info(out2);
        }

        var reqFile = new File(projectRoot, "requirements.txt");
        if (reqFile.exists()) {
            log.info("Installing requirements from {}...", reqFile.getAbsolutePath());
            var out3 = runCommand(List.of("uv", "pip", "install", "-r", "requirements.txt"), new File(projectRoot));
            log.info(out3);
        } else {
            log.error("requirements.txt not found in {}", projectRoot);
        }
    }

    private String runCommand(List<String> command) {
        return runCommand(command, new File(System.getProperty("user.dir")));
    }

    @SneakyThrows
    private String runCommand(List<String> command, File workingDir) {
        var pb = new ProcessBuilder(command);
        pb.directory(workingDir);
        pb.redirectErrorStream(true);
        var process = pb.start();

        var output = new StringBuilder();
        try (var reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Command failed with exit code " + exitCode + ": " + String.join(" ", command) + "\nOutput: " + output);
        }

        return output.toString();
    }

    /**
     * Starts the Gymnasium server with the specified parameters.
     */
    public void start() throws IOException {
        var cmd = new ArrayList<String>();
        cmd.add(pythonPath);
        cmd.add("-u"); // -u informs python to not buffer the stdout
        cmd.add(Paths.get(projectRoot, scriptPath).toString());
        cmd.add("--port");
        cmd.add(String.valueOf(port));
        cmd.add("--timeout");
        cmd.add(String.valueOf(timeout));
        cmd.add("--debug");
        cmd.add(String.valueOf(debug));
        cmd.add("--env_name");
        cmd.add(envName);

        if (envParams != null && !envParams.isEmpty()) {
            cmd.add("--env_params");
            cmd.add(GSON.toJson(envParams).replace("\"", "\\\""));
        }

        log.info("Executing command: {}", String.join(" ", cmd));
        log.info("Working Directory: {}", projectRoot);
        log.info("Python Path exists: {}", new File(pythonPath).exists());

        var pb = new ProcessBuilder(cmd);
        pb.redirectError(ProcessBuilder.Redirect.to(new File("python_debug.log")));
        pb.inheritIO();
        this.serverProcess = pb.start();

        try {
            Thread.sleep(WAIT_START);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void close() {
        try {
            if (serverProcess != null && serverProcess.isAlive()) {
                serverProcess.destroy();
                serverProcess.waitFor(1, TimeUnit.SECONDS);
                serverProcess.destroyForcibly();
            }
        } catch (InterruptedException ex) {
            if (serverProcess != null) serverProcess.destroyForcibly();
        }
    }

    static void main() {
        try (var launcher = new GymPythonLauncher(
                "env_server.py",
                5555,
                5000,
                true,
                "CartPole-v1",
                Map.of()
        )) {
            launcher.setupEnvironment();
            launcher.start();
            log.info("[+] Python Gymnasium server started via Java...");

            // Keep alive for testing
            Thread.sleep(15000);
        } catch (Exception e) {
            log.error("Error in communication between Java and Python gymnasium:", e);
        }
    }
}
