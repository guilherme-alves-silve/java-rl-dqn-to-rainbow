package br.com.guialves.rflr.utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;

public class PreProcessingWrapper {

    private final IEnv env;
    private final int skip;
    private final int resize;
    private final int concatenate;
    private final Image.Interpolation interpolation;

    public PreProcessingWrapper(IEnv env, int skip, int resize, int concatenate) {
        this(env, skip, resize, concatenate, Image.Interpolation.BILINEAR);
    }

    public PreProcessingWrapper(IEnv env, int skip, int resize, int concatenate, Image.Interpolation interpolation) {
        this.env = env;
        this.skip = skip;
        this.concatenate = concatenate;
        this.resize = resize;
        this.interpolation = interpolation;
    }

    public EnvStepResult step(ActionResult action) {
        var parent = env.manager();
        try (var sub = parent.newSubManager()) {
            var frames = new ArrayList<NDArray>();
            var rewards = new ArrayList<Double>();

            boolean term = false;
            boolean trunc = false;
            Map<Object, Object> info = null;

            for (int i = 0; i < concatenate; i++) {
                var skipResult = skipFrames(action, sub);

                try (NDArray grayState = grayscaleFrame(skipResult.state())) {
                    NDArray resizedState = resizeFrame(grayState);
                    frames.add(resizedState);
                }

                rewards.add(skipResult.reward());
                term  = skipResult.term();
                trunc = skipResult.trunc();
                info  = skipResult.info();

                if (term || trunc) {
                    break;
                }
            }

            var result = finishConcatenateFrames(frames, rewards);
            NDArray state = result.getKey();
            double totalReward = result.getValue();

            // Promote the surviving array out of sub before sub closes
            state.attach(parent);
            return new EnvStepResult(totalReward, term, trunc, info, state);
        }
    }

    private EnvStepResult skipFrames(ActionResult action, NDManager sub) {
        EnvStepResult stepResult = null;
        for (int i = 0; i < skip; i++) {
            stepResult = env.step(action, sub);
            if (stepResult.done()) {
                break;
            }
        }
        return stepResult;
    }

    private NDArray grayscaleFrame(NDArray state) {
        try (NDArray f32    = state.toType(DataType.FLOAT32, false);
             NDArray mean   = f32.mean(new int[]{2}, true);
             NDArray transp = mean.transpose(2, 0, 1)) {
            return transp.toType(DataType.UINT8, false);
        }
    }

    private NDArray resizeFrame(NDArray state) {
        try (NDArray expanded = state.expandDims(0);
             NDArray resized  = NDImageUtils.resize(expanded, resize, resize, interpolation)) {
            return resized.squeeze(0);
        }
    }

    private Pair<NDArray, Double> finishConcatenateFrames(List<NDArray> frames, List<Double> rewards) {
        while (frames.size() < concatenate) {
            frames.add(frames.getLast().duplicate());
            rewards.add(rewards.getLast());
        }

        NDArray state;
        try (var frameList = new NDList(frames)) {
            state = NDArrays.concat(frameList, 0);
        }

        double totalReward = rewards.stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        return new Pair<>(state, totalReward);
    }

    public Pair<NDArray, Map<Object, Object>> reset() {
        var parent = env.manager();
        var resetResult = env.reset();
        var info = resetResult.getKey();
        try (NDManager sub = parent.newSubManager()) {
            NDArray rawState = resetResult.getValue();

            try (NDArray gray    = grayscaleFrame(rawState);
                 NDArray resized = resizeFrame(gray)) {

                var frames  = new ArrayList<NDArray>();
                var rewards = new ArrayList<Double>();

                frames.add(resized.duplicate());
                rewards.add(0.0);

                var result = finishConcatenateFrames(frames, rewards);
                NDArray state = result.getKey();

                // Promote survivor out of sub before sub closes
                state.attach(parent);
                return new Pair<>(state, info);
            }
        }
    }
}
