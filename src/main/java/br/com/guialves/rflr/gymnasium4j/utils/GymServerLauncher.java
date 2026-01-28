package br.com.guialves.rflr.gymnasium4j.utils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class GymServerLauncher implements AutoCloseable {

    private Process serverProcess;
    private final String pythonPath;

    public GymServerLauncher() {
        String projectRoot = System.getProperty("user.dir");
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");

        this.pythonPath = isWindows
                ? Paths.get(projectRoot, ".venv", "Scripts", "python.exe").toString()
                : Paths.get(projectRoot, ".venv", "bin", "python").toString();
    }

    public void start(String scriptPath, int port) throws IOException {
        var pb = new ProcessBuilder(
                pythonPath,
                scriptPath,
                "--port", String.valueOf(port)
        );

        pb.inheritIO();
        this.serverProcess = pb.start();

        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void close() {
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroy();
            try {
                serverProcess.waitFor();
            } catch (InterruptedException e) {
                serverProcess.destroyForcibly();
            }
        }
    }

    static void main(String[] args) {
        try (GymServerLauncher launcher = new GymServerLauncher()) {
            launcher.start("python/gym_server.py", 5555);

            // Your DQN training code here
            Thread.sleep(10000);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}