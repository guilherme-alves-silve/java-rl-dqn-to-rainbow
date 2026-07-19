package br.com.guialves.rflr.fixture;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedExperience;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import lombok.experimental.UtilityClass;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@UtilityClass
public class ExperienceFixture {

    public static final Shape STATE_SHAPE = new Shape(3, 3);
    public static final Shape BATCH_1_SHAPE = new Shape(-1, 1);

    public static Experience createRandomExperience(NDManager manager, int i) {
        var state = manager.randomUniform(0, 10, STATE_SHAPE);
        var action = mock(ActionSpaceType.ActionResult.class);
        when(action.valueAs(Long.class)).thenReturn((long) i);
        var reward = -5 + (i + 1);
        var nextState = manager.randomNormal(STATE_SHAPE);
        boolean done = i % 2 == 0;
        return new Experience(state, action, reward, nextState, done);
    }

    public static PrioritizedExperience createRandomPriorityExperience(NDManager manager,
                                                                       int i,
                                                                       float priority) {
        var state = manager.randomUniform(0, 10, STATE_SHAPE);
        var action = mock(ActionSpaceType.ActionResult.class);
        when(action.valueAs(Long.class)).thenReturn((long) i);
        var reward = -5 + (i + 1);
        var nextState = manager.randomNormal(STATE_SHAPE);
        boolean done = i % 2 == 0;
        return new PrioritizedExperience(state, action, reward, nextState, done, priority);
    }
}
