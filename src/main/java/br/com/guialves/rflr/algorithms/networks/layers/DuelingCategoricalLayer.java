package br.com.guialves.rflr.algorithms.networks.layers;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.function.Consumer;

import static br.com.guialves.rflr.djlutils.DJLUtils.N_BATCH;

public class DuelingCategoricalLayer extends DuelingLayer {

    private final int atoms;

    public DuelingCategoricalLayer(int actions,
                                   int atoms,
                                   DuelingType duelingType,
                                   Consumer<SequentialBlock> featureBackboneModifier,
                                   Consumer<SequentialBlock> valueHeadModifier,
                                   Consumer<SequentialBlock> advantageHeadModifier) {
        super(actions, duelingType, featureBackboneModifier, valueHeadModifier, advantageHeadModifier);
        this.atoms = atoms;
    }

    @Override
    public NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        if (inputs.size() != 1) throw new IllegalArgumentException("The dueling categorical dqn just accepts one input!");
        var feature = featureBackbone.forward(parameterStore, inputs, training);
        // (batch, 1, atoms)
        var value = valueHead.forward(parameterStore, feature, training).singletonOrThrow()
                .reshape(N_BATCH, 1, atoms);
        // (batch, actions, atoms)
        var advantage = advantageHead.forward(parameterStore, feature, training).singletonOrThrow()
                .reshape(N_BATCH, actions, atoms);
        // (batch, actions, atoms)
        return new NDList(qValueCalc.apply(value, advantage));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {new Shape(inputShapes[0].get(0), actions, atoms)};
    }
}
