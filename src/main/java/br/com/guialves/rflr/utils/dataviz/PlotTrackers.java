package br.com.guialves.rflr.utils.dataviz;

import me.tongfei.progressbar.ProgressBar;

import java.util.ArrayList;
import java.util.List;

public class PlotTrackers {

    private final List<Float> episodeEpsilons;
    private final List<Float> meanEpisodeRewards;
    private final List<Float> meanEpisodeLoss;

    public PlotTrackers() {
        this.episodeEpsilons = new ArrayList<>();
        this.meanEpisodeRewards = new ArrayList<>();
        this.meanEpisodeLoss = new ArrayList<>();
    }

    public void setTrackersMessage(ProgressBar pbar, int frames, int bufferSize) {
        if (!episodeEpsilons.isEmpty() &&
                !meanEpisodeRewards.isEmpty() &&
                !meanEpisodeLoss.isEmpty()) {
            pbar.setExtraMessage(String.format(
                    "ε=%.4f 🪙=%.4f 📉=%.4f 🖼️=%d, 📥=%d",
                    episodeEpsilons.getLast(),
                    meanEpisodeRewards.getLast(),
                    meanEpisodeLoss.getLast(),
                    frames,
                    bufferSize
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
        IO.println("ε=epsilon decay, 🪙=reward, 📉=loss, 🖼️=frames, 📥=buffer size");
    }
}
