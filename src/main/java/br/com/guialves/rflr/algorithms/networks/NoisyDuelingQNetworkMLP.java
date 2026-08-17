package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingLayer;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer.noisyLayer;
import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.copy;
import static br.com.guialves.rflr.djlutils.DJLUtils.initGradients;

/**
 * After each update, it's recommended that you call the method {@code resetNoise()}
 * Reference:
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05_noisy_net.py">Python - Noisy Net</a>
 */
@Slf4j
@Accessors(fluent = true)
public class NoisyDuelingQNetworkMLP implements INoisyNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final Model model;
    private final DuelingLayer net;
    @Getter
    private final DuelingType duelingType;
    private final List<NoisyLayer> noisyLayers;
    private final ParameterStore parameterStore;

    public static NoisyDuelingQNetworkMLP withMeanType(int observations,
                                                       int actions,
                                                       NDManager parent) {
        return new NoisyDuelingQNetworkMLP(observations, actions, null, null, parent, DuelingType.MEAN);
    }

    public static NoisyDuelingQNetworkMLP withMaxType(int observations,
                                                      int actions,
                                                      NDManager parent) {
        return new NoisyDuelingQNetworkMLP(observations, actions, null, null, parent, DuelingType.MAX);
    }

    public NoisyDuelingQNetworkMLP(int observations,
                                   int actions,
                                   NDManager parent,
                                   DuelingType duelingType) {
        this(observations, actions, null, null, parent, duelingType);
    }

    @SneakyThrows
    public NoisyDuelingQNetworkMLP(int observations,
                                   int actions,
                                   Path modelPath,
                                   String prefix,
                                   NDManager parent,
                                   DuelingType duelingType) {
        this.observations = observations;
        this.actions = actions;
        this.subManager = subMgr(parent, getClass());
        this.duelingType = duelingType;
        this.noisyLayers = new ArrayList<>();
        this.model = newModel(getClass(), subManager.getDevice());
        this.net = new DuelingLayer(
                actions,
                duelingType,
                featureBackbone ->
                        featureBackbone.add(linear(128))
                            .add(Activation::relu)
                            .add(addAndGet(noisyLayers, noisyLayer(128)))
                            .add(Activation::relu)
                            .add(addAndGet(noisyLayers, noisyLayer(128)))
                            .add(Activation::relu),
                valueHead ->
                        valueHead.add(addAndGet(noisyLayers, noisyLayer(1))),
                advantageHead ->
                        advantageHead.add(addAndGet(noisyLayers, noisyLayer(actions)))
        );
        model.setBlock(net);

        this.parameterStore = new ParameterStore(subManager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(subManager, DataType.FLOAT32, new Shape(1, observations));
            initGradients(model.getBlock());
            this.training = true;
        }
    }

    @Override
    public void resetNoise() {
        noisyLayers.forEach(NoisyLayer::resetNoise);
    }

    @Override
    public NDList forward(NDList input) {
        return safeForwardSingle(subManager, net, parameterStore, input, training);
    }

    @Override
    public NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    @Override
    @SneakyThrows
    public void save(Path modelPath, String newModelName) {
        this.model.save(modelPath, newModelName);
    }

    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    @Override
    public void eval() {
        this.training = false;
    }

    @Override
    public void train() {
        this.training = true;
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public IDeepQNetwork clone() {
        var cloned = new NoisyDuelingQNetworkMLP(observations, actions, subManager.getParentManager(), duelingType);
        setName(cloned.subManager, "clone");
        copy(model.getBlock(), cloned.model.getBlock());
        return cloned;
    }

    @Override
    public NDManager subManager() {
        return subManager;
    }

    @Override
    public void close() {
        subManager.close();
        model.close();
        resetNoise();
    }
}
