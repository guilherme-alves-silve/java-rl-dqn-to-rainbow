package br.com.guialves.rflr.execs;

import br.com.guialves.rflr.fixture.AgentDQNResourceFixture;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentExecsMainTest {

    private static final String FRAMES_LIMIT = "1000";
    private static final String BUFFER_CAPACITY = "128";
    private static final String DONT_SAVE_MODEL = "false";
    private static final String DONT_RECORD =  "false";
    private static final String DONT_SHOW_METRICS = "false";
    private static final String DONT_RENDER = "false";
    private static final String DONT_SAVE_CONFIG = "false";

    @BeforeEach
    public void setUpEach() {
        System.setProperty("agent.framesLimit", FRAMES_LIMIT);
        System.setProperty("agent.bufferCapacity", BUFFER_CAPACITY);
        System.setProperty("agent.saveModel", DONT_SAVE_MODEL);
        System.setProperty("agent.records", DONT_RECORD);
        System.setProperty("agent.showAllMetrics", DONT_SHOW_METRICS);
        System.setProperty("agent.renderRun", DONT_RENDER);
        System.setProperty("agent.saveConfig", DONT_SAVE_CONFIG);
    }

    @Test
    void shouldRunAgentDQNMain() {
        assertDoesNotThrow((Executable) AgentDQNMain::main);
    }

    @Test
    void shouldRunAgentDDQNMain() {
        assertDoesNotThrow((Executable) AgentDDQNMain::main);
    }

    @Test
    void shouldRunAgentDQNPERMain() {
        assertDoesNotThrow((Executable) AgentDQNPERMain::main);
    }

    @Test
    void shouldRunAgentDuelingDQNMain() {
        assertDoesNotThrow((Executable) AgentDuelingDQNMain::main);
    }

    @Test
    void shouldRunAgentNStepDQNMain() {
        assertDoesNotThrow((Executable) AgentNStepDQNMain::main);
    }

    @Test
    void shouldRunAgentNoisyNetDQNMain() {
        assertDoesNotThrow((Executable) AgentNoisyNetDQNMain::main);
    }

    @Test
    void shouldRunAgentNoisyDuelingNetDDQNMain() {
        assertDoesNotThrow((Executable) AgentNoisyDuelingNetDDQNMain::main);
    }

    @Test
    void shouldRunAgentC51DQNMain() {
        assertDoesNotThrow((Executable) AgentC51DQNMain::main);
    }

    @Test
    void shouldRunAgentC51NoisyNetDQNPERMain() {
        assertDoesNotThrow((Executable) AgentC51NoisyNetDQNPERMain::main);
    }

    @Test
    void shouldRunAgentC51RReLUDQNMain() {
        assertDoesNotThrow((Executable) AgentC51RReLUDQNMain::main);
    }

    @Test
    void shouldRunAgentRainbowDQNMain() {
        assertDoesNotThrow((Executable) AgentRainbowDQNMain::main);
    }

    /**
     * End-to-end test for the {@code --load-model} flow on {@link AgentDQNMain}:
     * parse the args, copy the committed fixture {@code .params} into the
     * location the runner reads from
     * ({@code ./output_models/dqn/}), run the agent, and verify no exception
     * is raised while training is skipped and the saved network is replayed.
     */
    @Test
    @DisplayName("AgentDQNMain: --load-model loads from resources and runs without training")
    void shouldLoadModelAndRunOnly() throws Exception {
        assertTrue(Files.exists(AgentDQNResourceFixture.PARAMS_FILE),
                "Missing fixture at " + AgentDQNResourceFixture.PARAMS_FILE
                        + " - run AgentDQNResourceFixture#main once and commit the file.");

        var targetDir = Path.of("./output_models/dqn");
        Files.createDirectories(targetDir);
        var targetFile = targetDir.resolve(
                AgentDQNResourceFixture.MODEL_NAME + "-0000.params");
        Files.copy(AgentDQNResourceFixture.PARAMS_FILE, targetFile,
                StandardCopyOption.REPLACE_EXISTING);

        try {
            assertDoesNotThrow(() -> AgentDQNMain.main(new String[] {
                    "--load-model", AgentDQNResourceFixture.MODEL_NAME
            }));
        } finally {
            Files.deleteIfExists(targetFile);
        }
    }
}
