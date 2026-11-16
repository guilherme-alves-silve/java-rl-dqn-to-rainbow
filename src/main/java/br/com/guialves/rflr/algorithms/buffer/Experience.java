package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;
import lombok.Getter;
import lombok.experimental.Accessors;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.duplicateToCpu;

/**
 * A single reinforcement-learning transition {@code (s, a, r, s', done)}
 * produced by one environment step.
 *
 * <p>{@code state} and {@code nextState} are always kept on the CPU.
 * Transitions are stored on the CPU and only moved to the GPU in batches
 * at training time, since keeping thousands of individual transitions in
 * VRAM would fragment it without a throughput benefit.</p>
 *
 * <p>The constructor <b>copies</b> the arrays it receives, so this instance
 * owns its own {@code state}, {@code action} and {@code nextState}. The
 * caller keeps ownership of the originals and must call {@link #close()}
 * on the {@code Experience} (not on the arrays it passed in) when done.</p>
 */
@Getter
@Accessors(fluent = true)
public class Experience implements AutoCloseable {

    private final int envNumber;
    private final NDArray state;
    private final ActionResult action;
    private final double reward;
    private final NDArray nextState;
    private final boolean term;
    private final boolean trunc;

    /** Default environment index for single-environment setups. */
    public static final int DEFAULT_ENV_NUMBER = 0;

    /**
     * Creates a transition for a single environment
     * ({@code envNumber = DEFAULT_ENV_NUMBER}).
     */
    public static Experience copy(NDArray state, ActionResult action, double reward, NDArray nextState, boolean term, boolean trunc) {
        return new Experience(DEFAULT_ENV_NUMBER, state, action, reward, nextState, term, trunc);
    }

    public static Experience copy(int envNumber, NDArray state, ActionResult action, double reward, NDArray nextState, boolean term, boolean trunc) {
        return new Experience(envNumber, state, action, reward, nextState, term, trunc);
    }

    /**
     * Creates a transition for a specific environment.
     * The given {@code state}, {@code action} and {@code nextState} are copied.
     *
     * @throws NullPointerException if {@code state}, {@code action} or
     *         {@code nextState} is {@code null}.
     */
    private Experience(int envNumber, NDArray state, ActionResult action,
                       double reward, NDArray nextState, boolean term, boolean trunc) {
        this.envNumber = envNumber;
        this.state = duplicateToCpu(state);
        this.action = action.duplicate();
        this.reward = reward;
        this.nextState = duplicateToCpu(nextState);
        this.term = term;
        this.trunc = trunc;
    }

    /**
     * done if it's term or trunc (terminated or truncated)
     * terminated: When the episode is done in the MDP (no future)
     */
    public boolean done() {
        return term || trunc;
    }

    /**
     * Returns the action value cast to {@code clazz}, e.g.
     * {@code actionAs(Integer.class)} or {@code actionAs(float[].class)}.
     */
    public <T> T actionAs(Class<T> clazz) {
        return action.valueAs(clazz);
    }

    /**
     * Releases the native memory of {@code state}, {@code action} and
     * {@code nextState}. Do not use this instance afterward.
     */
    @Override
    public void close() {
        state.close();
        action.close();
        nextState.close();
    }
}
