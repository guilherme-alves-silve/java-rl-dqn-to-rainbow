package br.com.guialves.rflr.utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.Env;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PreProcessingWrapper {

    private final Env env;
    private final int skip;
    private final int resize;
    private final int concatenate;
    private final Image.Interpolation interpolation;

    public PreProcessingWrapper(Env env, int skip, int resize, int concatenate) {
        this(env, skip, resize, concatenate, Image.Interpolation.BILINEAR);
    }

    public PreProcessingWrapper(Env env, int skip, int resize, int concatenate, Image.Interpolation interpolation) {
        this.env = env;
        this.skip = skip;
        this.concatenate = concatenate;
        this.resize = resize;
        this.interpolation = interpolation;
    }

    public EnvStepResult step(int action) {
        var frames = new ArrayList<NDArray>();
        var rewards = new ArrayList<Double>();

        boolean term = false;
        boolean trunc = false;
        Map<String, Object> info = null;

        for (int i = 0; i < concatenate; i++) {
            var skipResult = skipFrames(action);
            var grayState = grayscaleFrame(skipResult.state());
            var resizedState = resizeFrame(grayState);

            frames.add(resizedState);
            rewards.add(skipResult.reward());

            term = skipResult.term();
            trunc = skipResult.trunc();
            info = skipResult.info();

            if (term || trunc) {
                break;
            }
        }

        var result = finishConcatenateFrames(frames, rewards);
        return new EnvStepResult(result.getValue(), term, trunc, info, result.getKey());
    }

    private EnvStepResult skipFrames(int action) {
        EnvStepResult stepResult = null;
        for (int i = 0; i <= skip; i++) {
            stepResult = env.step(action);
            if (stepResult.done()) {
                break;
            }
        }

        return stepResult;
    }

    private NDArray grayscaleFrame(NDArray state) {
        state = state.toType(DataType.FLOAT32, false);

        var grayState = state.mean(new int[]{2}, true);

        // [H, W, 1] -> [1, H, W]
        grayState = grayState.transpose(2, 0, 1);

        grayState = grayState.toType(DataType.UINT8, false);

        return grayState;
    }

    private NDArray resizeFrame(NDArray state) {
        // [1, H, W] -> [1, 1, H, W]
        var resized = NDImageUtils.resize(state.expandDims(0), resize, resize, interpolation);
        // [1, 1, H, W] -> [1, H, W]
        return resized.squeeze(0);
    }

    private Pair<NDArray, Double> finishConcatenateFrames(List<NDArray> frames, List<Double> rewards) {

        while (frames.size() < concatenate) {
            var lastState = frames.getLast();
            var lastReward = rewards.getLast();

            frames.add(lastState.duplicate());
            rewards.add(lastReward);
        }

        var state = NDArrays.concat(new NDList(frames), 0);

        double totalReward = rewards.stream()
                .mapToDouble(Double::doubleValue)
                .sum();
        return new Pair<>(state, totalReward);
    }

    public Pair<NDArray, Map<Object, Object>> reset() {
        var resetResult = env.reset();
        var state = resetResult.getValue();
        var info = resetResult.getKey();

        var frames = new ArrayList<NDArray>();
        var rewards = new ArrayList<Double>();

        var grayState = grayscaleFrame(state);
        var resizedState = resizeFrame(grayState);

        frames.add(resizedState);
        rewards.add(0.0);

        var result = finishConcatenateFrames(frames, rewards);

        return new Pair<>(result.getKey(), info);
    }
}
