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
    private final String algorithmName;
    private int lastResourcesCount;
    private int lastResourcesCountPerEpisode;

    public PlotTrackers(String algorithmName) {
        this(100, algorithmName);
    }

    public PlotTrackers(int framesToCalcMemResources, String algorithmName) {
        this.episodeEpsilons = new ArrayList<>();
        this.meanEpisodeRewards = new ArrayList<>();
        this.meanEpisodeLoss = new ArrayList<>();
        this.framesToCalcMemResources = framesToCalcMemResources;
        this.algorithmName = algorithmName;
    }

    public void updateTrackersMessage(ProgressBar pbar,
                                      int frames,
                                      int bufferSize,
                                      NDManager parent) {
        updateTrackersMessage(pbar, frames, bufferSize, parent, null);
    }

    public void updateTrackersMessage(ProgressBar pbar,
                                      int frames,
                                      int bufferSize,
                                      NDManager parent,
                                      NDManager parentPerEpisode) {
        if (!episodeEpsilons.isEmpty() &&
                !meanEpisodeRewards.isEmpty() &&
                !meanEpisodeLoss.isEmpty()) {

            if (frames % framesToCalcMemResources == 0) {
                lastResourcesCount = managedArrayCount(parent);
                lastResourcesCountPerEpisode = null == parentPerEpisode? 0 : managedArrayCount(parentPerEpisode);
            }

            pbar.setExtraMessage(String.format(
                    "ε=%.4f 🪙=%.4f 📉=%.4f 🖼️=%d, 📥=%d, res=(all=%d, epi=%d)",
                    episodeEpsilons.getLast(),
                    meanEpisodeRewards.getLast(),
                    meanEpisodeLoss.getLast(),
                    frames,
                    bufferSize,
                    lastResourcesCount,
                    lastResourcesCountPerEpisode
            ));
        }
    }

    public void showAllMetrics() {
        PlotTrackersTablesaw.showAllMetrics(episodeEpsilons, meanEpisodeRewards, meanEpisodeLoss, algorithmName);
        episodeEpsilons.clear();
        meanEpisodeRewards.clear();
        meanEpisodeLoss.clear();
    }

    public void add(float epsilon, List<Double> episodeRewards, float avgLoss) {
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
