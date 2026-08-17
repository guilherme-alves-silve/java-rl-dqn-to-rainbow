package br.com.guialves.rflr.execs;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

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
        assertDoesNotThrow(AgentDQNMain::main);
    }

    @Test
    void shouldRunAgentDDQNMain() {
        assertDoesNotThrow(AgentDDQNMain::main);
    }

    @Test
    void shouldRunAgentDQNPERMain() {
        assertDoesNotThrow(AgentDQNPERMain::main);
    }

    @Test
    void shouldRunAgentDuelingDQNMain() {
        assertDoesNotThrow(AgentDuelingDQNMain::main);
    }

    @Test
    void shouldRunAgentNStepDQNMain() {
        assertDoesNotThrow(AgentNStepDQNMain::main);
    }

    @Test
    void shouldRunAgentNoisyNetDQNMain() {
        assertDoesNotThrow(AgentNoisyNetDQNMain::main);
    }

    @Test
    void shouldRunAgentNoisyDuelingNetDDQNMain() {
        assertDoesNotThrow(AgentNoisyDuelingNetDDQNMain::main);
    }

    @Test
    void shouldRunAgentC51DQNMain() {
        assertDoesNotThrow(AgentC51DQNMain::main);
    }
}
