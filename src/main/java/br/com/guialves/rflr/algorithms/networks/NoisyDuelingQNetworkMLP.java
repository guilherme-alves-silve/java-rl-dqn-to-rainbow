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
import br.com.guialves.rflr.djlutils.DJLUtils;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer.noisyLayer;
import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.safeForwardSingle;

/**
 * After each update, it's recommended that you call the method {@code resetNoise()}
 * Reference:
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05_noisy_net.py">Python - Noisy Net</a>
 */
@Slf4j
public class NoisyDuelingQNetworkMLP implements IDeepQNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final Model model;
    private final DuelingLayer net;
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
        this.subManager = parent.newSubManager();
        this.duelingType = duelingType;
        this.noisyLayers = new ArrayList<>();
        this.model = Model.newInstance("noisy_dueling_net_dqn_mlp", subManager.getDevice());
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
            DJLUtils.setGradients(model.getBlock());
            this.training = true;
        }
    }

    private Block addAndGet(List<NoisyLayer> list, NoisyLayer noisyLayer) {
        list.add(noisyLayer);
        return noisyLayer;
    }

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
    public IDeepQNetwork clone() {
        var cloned = new NoisyDuelingQNetworkMLP(observations, actions, subManager, duelingType);
        DJLUtils.copy(model.getBlock(), cloned.model.getBlock());
        return cloned;
    }

    @Override
    public NDManager manager() {
        return this.subManager;
    }

    @Override
    public void close() {
        model.close();
    }
}
