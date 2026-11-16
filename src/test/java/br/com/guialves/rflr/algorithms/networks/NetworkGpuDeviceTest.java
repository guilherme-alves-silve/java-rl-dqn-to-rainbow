package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLUtils.gpuCount;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * When a GPU is available, the networks (and their parameters) must be allocated
 * on the GPU and never on the CPU.
 */
class NetworkGpuDeviceTest {

    private static final int OBSERVATIONS = 4;
    private static final int CNN_OBSERVATIONS = 84;
    private static final int ACTIONS = 6;

    @Test
    void shouldAllocateTheMlpNetworkOnGpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            try (var net = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, manager)) {
                assertEquals(Device.gpu(), net.subManager().getDevice());
                assertParametersOnGpu(net);

                var input = manager.create(new Shape(1, OBSERVATIONS));
                try (var output = net.forward(input)) {
                    assertEquals(Device.gpu(), output.getDevice());
                }
            }
        }
    }

    @Test
    void shouldAllocateTheCnnNetworkOnGpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        int channels = 4;
        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            try (var net = new DeepQNetworkCNN(channels, CNN_OBSERVATIONS, ACTIONS, manager)) {
                assertEquals(Device.gpu(), net.subManager().getDevice());
                assertParametersOnGpu(net);

                var input = manager.create(new Shape(1, channels, CNN_OBSERVATIONS, CNN_OBSERVATIONS));
                try (var output = net.forward(input)) {
                    assertEquals(Device.gpu(), output.getDevice());
                }
            }
        }
    }

    @Test
    void shouldAllocateTheRainbowNetworkOnGpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            try (var net = new RainbowQNetworkMLP(OBSERVATIONS, ACTIONS, manager, DuelingType.MEAN)) {
                assertEquals(Device.gpu(), net.subManager().getDevice());
                assertParametersOnGpu(net);
            }
        }
    }

    @Test
    void shouldAllocateTheNetworkOnGpuEvenWhenTheParentIsOnTheCpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        // Device policy: everything that is computed lives in the GPU, even when the
        // parent manager is on the CPU.
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            try (var net = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, manager)) {
                assertEquals(Device.gpu(), net.subManager().getDevice());
                assertParametersOnGpu(net);
            }
        }
    }

    @Test
    void shouldAllocateTheNetworkOnCpuWhenThereIsNoGpu() {
        assumeTrue(gpuCount() == 0, "GPU is available");

        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            try (var net = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, manager)) {
                assertEquals(Device.cpu(), net.subManager().getDevice());
                assertParametersOnCpu(net);
            }
        }
    }

    private void assertParametersOnGpu(IDeepQNetwork net) {
        var parameters = net.getBlock().getParameters();
        for (int i = 0; i < parameters.size(); ++i) {
            assertEquals(Device.gpu(), parameters.valueAt(i).getArray().getDevice(),
                    "the parameter " + parameters.keyAt(i) + " must be on the GPU");
        }
    }

    private void assertParametersOnCpu(IDeepQNetwork net) {
        var parameters = net.getBlock().getParameters();
        for (int i = 0; i < parameters.size(); ++i) {
            assertEquals(Device.cpu(), parameters.valueAt(i).getArray().getDevice(),
                    "the parameter " + parameters.keyAt(i) + " must be on the CPU");
        }
    }
}
