package br.com.guialves.rflr.integration.stableretro;

import org.junit.jupiter.api.Test;
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.BindMode;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.DockerImageName;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs the Stable Retro integration tests <b>inside a Linux container</b>.
 *
 * <p>Why: the {@code stable_retro} Python package has a compiled C++ core and only ships
 * manylinux wheels. On Windows it must be built from source (cmake + a C++ compiler), which
 * is why {@code StableRetroGymTest#shouldBuildRetroVectorEnvFromEnvName} is skipped there
 * ({@code assumeFalse(onWindows)}).</p>
 *
 * <p>Because the extension is a Linux binary, it is not enough to have the site-packages:
 * the <b>test itself</b> has to run inside the container, where the embedded CPython finds
 * the Python packages through {@code JAVA_RL_SITE_PACKAGES}.</p>
 *
 * <p>It uses the <b>same image</b> of the agent ({@code docker/Dockerfile}): the image carries
 * the environment (JDK, Python, the Python packages with the compiled stable-retro core and
 * the Maven dependencies) and the code is mounted as a volume.</p>
 *
 * <pre>
 * docker build -t java-rl-dqn-to-rainbow:latest .
 * </pre>
 *
 * <p>Without a Docker daemon the test is skipped (it is an environment test, not a unit
 * test).</p>
 */
class StableRetroContainerTest {

    private static final String IMAGE = "java-rl-dqn-to-rainbow:latest";
    private static final Duration BUILD_TIMEOUT = Duration.ofMinutes(20);

    @Test
    void shouldRunTheStableRetroIntegrationTestsInsideTheContainer() throws Exception {
        assumeTrue(DockerClientFactory.instance().isDockerAvailable(),
                "Docker is not available: the stable-retro tests need a Linux container");

        var projectDir = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        var reports = Files.createTempDirectory("retro-surefire");

        try (var container = new GenericContainer<>(DockerImageName.parse(IMAGE))
                .withCommand("mvn", "-B", "-o",
                        "-Dgroups=integration", "-DexcludedGroups=",
                        "-Dtest=StableRetroEnvIT", "test")
                .withFileSystemBind(projectDir.toString(), "/build", BindMode.READ_WRITE)
                .withFileSystemBind(reports.toString(), "/build/target/surefire-reports", BindMode.READ_WRITE)
                .withEnv("JAVA_RL_SITE_PACKAGES", "/opt/venv/lib/python3.12/site-packages")
                .waitingFor(Wait.forLogMessage("(?s).*BUILD (SUCCESS|FAILURE).*", 1)
                        .withStartupTimeout(BUILD_TIMEOUT))) {

            container.start();

            var exitCode = container.getContainerInfo().getState().getExitCodeLong();
            var report = reports.resolve("TEST-br.com.guialves.rflr.gymnasium4j.StableRetroEnvIT.xml");

            if (exitCode != 0) {
                var details = Files.exists(report) ? Files.readString(report) : "(no surefire report)";
                fail("The stable-retro integration tests failed inside the container (exit code "
                        + exitCode + "):\n" + details);
            }

            assertTrue(Files.exists(report),
                    "The surefire report must be produced inside the container: " + report);
            var content = Files.readString(report);
            assertTrue(content.contains("tests=\"2\""),
                    "The 2 integration tests must have run inside the container:\n" + content);
            assertFalse(content.contains("skipped=\"2\""),
                    "The tests must not be skipped inside the container:\n" + content);
        }
    }
}
