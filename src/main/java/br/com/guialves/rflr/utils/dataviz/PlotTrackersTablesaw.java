package br.com.guialves.rflr.utils.dataviz;

import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

import java.util.List;

public class PlotTrackersTablesaw {

    private PlotTrackersTablesaw() {
        throw new AssertionError("No PlotTrackersTablesaw!");
    }

    public static void showAllMetrics(List<Float> epsilons, List<Float> rewards, List<Float> losses) {

        if (epsilons.isEmpty()) return;

        var table = Table.create("Training Metrics");
        table.addColumns(
                DoubleColumn.create("Episode", range(0, epsilons.size())),
                DoubleColumn.create("Epsilon", toDoubleArray(epsilons)),
                DoubleColumn.create("Reward", toDoubleArray(rewards)),
                DoubleColumn.create("Loss", toDoubleArray(losses))
        );

        Plot.show(LinePlot.create("Epsilon over Episodes", table, "Episode", "Epsilon"));
        Plot.show(LinePlot.create("Rewards over Episodes", table, "Episode", "Reward"));
        Plot.show(LinePlot.create("Loss over Episodes", table, "Episode", "Loss"));
    }

    private static double[] range(int start, int end) {
        var result = new double[end - start];
        for (int i = 0; i < result.length; i++) {
            result[i] = start + i;
        }
        return result;
    }

    private static double[] toDoubleArray(List<Float> list) {
        return list.stream()
                .mapToDouble(Float::doubleValue)
                .toArray();
    }
}
