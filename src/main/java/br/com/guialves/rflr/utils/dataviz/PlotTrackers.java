package br.com.guialves.rflr.utils.dataviz;

import ai.djl.ndarray.NDManager;
import me.tongfei.progressbar.ProgressBar;

import java.util.ArrayList;
import java.util.List;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;

public class PlotTrackers {

    private final List<Float> episodeEpsilons;
    private final List<Float> meanEpisodeRewards;
    private final List<Float> meanEpisodeLoss;
    private final int framesToCalcMemResources;
    private int lastResourcesCount;

    public PlotTrackers() {
        this(10_000);
    }

    public PlotTrackers(int framesToCalcMemResources) {
        this.episodeEpsilons = new ArrayList<>();
        this.meanEpisodeRewards = new ArrayList<>();
        this.meanEpisodeLoss = new ArrayList<>();
        this.framesToCalcMemResources = framesToCalcMemResources;
    }

    public void setTrackersMessage(ProgressBar pbar,
                                   int frames,
                                   int bufferSize,
                                   NDManager parent) {
        if (!episodeEpsilons.isEmpty() &&
                !meanEpisodeRewards.isEmpty() &&
                !meanEpisodeLoss.isEmpty()) {

            if (frames % framesToCalcMemResources == 0) {
                lastResourcesCount = managedArrayCount(parent);
            }

            pbar.setExtraMessage(String.format(
                    "ε=%.4f 🪙=%.4f 📉=%.4f 🖼️=%d, 📥=%d, res=%d",
                    episodeEpsilons.getLast(),
                    meanEpisodeRewards.getLast(),
                    meanEpisodeLoss.getLast(),
                    frames,
                    bufferSize,
                    lastResourcesCount
            ));
        }
    }

    public void showAllMetrics() {
        PlotTrackersTablesaw.showAllMetrics(episodeEpsilons, meanEpisodeRewards, meanEpisodeLoss);
    }

    public void add(float epsilon, ArrayList<Object> episodeRewards, float avgLoss) {
        float avgReward = (float) episodeRewards.stream()
                .mapToDouble(reward -> (double) reward)
                .average()
                .orElse(0.0);
        episodeEpsilons.add(epsilon);
        meanEpisodeRewards.add(avgReward);
        meanEpisodeLoss.add(avgLoss);
    }

    public void explainTrackersMessage() {
        IO.println("ε=epsilon decay, 🪙=reward, 📉=loss, 🖼️=frames, 📥=buffer size, res=alloc mem DJL");
    }
}
