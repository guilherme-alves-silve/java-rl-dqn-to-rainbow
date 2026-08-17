package br.com.guialves.rflr.algorithms.networks;

import ai.djl.nn.Block;
import br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer;

import java.util.List;

public interface INoisyNetwork extends IDeepQNetwork {

    void resetNoise();

    default Block addAndGet(List<NoisyLayer> list, NoisyLayer noisyLayer) {
        list.add(noisyLayer);
        return noisyLayer;
    }
}
