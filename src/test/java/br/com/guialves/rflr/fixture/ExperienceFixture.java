package br.com.guialves.rflr.fixture;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import lombok.Cleanup;
import lombok.experimental.UtilityClass;

import static br.com.guialves.rflr.algorithms.buffer.Experience.DEFAULT_ENV_NUMBER;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@UtilityClass
public class ExperienceFixture {

    public static final Shape STATE_SHAPE = new Shape(3, 3);
    public static final Shape BATCH_1_SHAPE = new Shape(-1, 1);

    public static Experience createRandomExperience(NDManager manager, int i) {
        return createRandomExperience(manager, DEFAULT_ENV_NUMBER, i);
    }

    public static Experience createRandomExperience(NDManager manager, int envNumber, int i) {
        @Cleanup var state = manager.randomUniform(0, 10, STATE_SHAPE);
        @Cleanup var action = mock(ActionSpaceType.ActionResult.class);
        when(action.valueAs(Long.class)).thenReturn((long) i);
        when(action.duplicate()).thenReturn(action);
        var reward = -5 + (i + 1);
        @Cleanup var nextState = manager.randomNormal(STATE_SHAPE);
        boolean done = i % 2 == 0;
        return Experience.copy(envNumber, state, action, reward, nextState, done, done);
    }
}
