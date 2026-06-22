package br.com.guialves.rflr.playground;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class PlaygroundMatrixOperationsTest {

    private static NDManager manager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterAll
    static void shutdown() {
        manager.close();
    }

    /**
     * You can find the visual explain in
     * this repository on the path: docs/graphics/print_idea_djl_gather_actions_edited_paint.jpg
     * Observation: This image was a print from the output of this test,
     * and it was sent to gemini (curated a few times) to make the pointers (then was edited in paint and sent again).
     * Reference:
     *  <a href="https://neuralpalace.substack.com/p/how-to-never-forget-deep-q-networks">...</a>
     */
    @Test
    void shouldCheckGather() {
        float[][] modelOutputData = {
                {0.0334f, 0.0188f, 0.0293f, 0.0255f, -0.0399f},
                {0.0339f, 0.0183f, 0.0293f, 0.0257f, -0.0396f},
                {0.0339f, 0.0183f, 0.0293f, 0.0257f, -0.0396f},
                {0.0329f, 0.0190f, 0.0294f, 0.0253f, -0.0390f},
                {0.0333f, 0.0187f, 0.0294f, 0.0257f, -0.0400f},
                {0.0335f, 0.0185f, 0.0294f, 0.0257f, -0.0400f}
        };

        var modelOutput = manager.create(modelOutputData);
        IO.println("modelOutput shape: " + modelOutput.getShape()); // (6, 5)

        long[] actionsData = {0, 1, 2, 3, 4, 1};

        var actions = manager.create(actionsData);
        IO.println("actions shape: " + actions.getShape()); // (6)
        var actionsExpanded = actions.expandDims(1);
        IO.println("actions expanded shape: " + actionsExpanded.getShape()); // (6)

        IO.println(modelOutput);
        var result = modelOutput.gather(actionsExpanded, 1); // axis=1

        IO.println("result shape: " + result.getShape()); // (6, 1)

        IO.println("Result of gather:");
        IO.println(result);
    }

    @Test
    void shouldCheckMaskDones() {

        var dones = manager.ones(new Shape(5), DataType.INT32);
        var notDones = manager.zeros(new Shape(5), DataType.INT32);

        var maskDones = dones.neg().add(1);
        var maskNotDones = notDones.neg().add(1);
        IO.println("Mask done: " + maskDones);
        IO.println("Mask not done: " + maskNotDones);

        float[][] modelOutputData = {
                {0.0334f, 0.0188f, 0.0293f, 0.0255f, -0.0399f},
                {0.0339f, 0.0183f, 0.0293f, 0.0257f, -0.0396f},
                {0.0339f, 0.0183f, 0.0293f, 0.0257f, -0.0396f},
                {0.0329f, 0.0190f, 0.0294f, 0.0253f, -0.0390f},
                {0.0333f, 0.0187f, 0.0294f, 0.0257f, -0.0400f},
                {0.0335f, 0.0185f, 0.0294f, 0.0257f, -0.0400f}
        };

        var modelOutput = manager.create(modelOutputData);

        IO.println(modelOutput.mul(maskDones));
        IO.println(modelOutput.mul(maskNotDones));
    }
}
