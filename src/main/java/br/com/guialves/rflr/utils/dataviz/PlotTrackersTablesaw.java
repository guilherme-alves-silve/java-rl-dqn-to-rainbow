package br.com.guialves.rflr.utils.dataviz;

import lombok.SneakyThrows;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

import java.io.File;
import java.nio.file.Files;
import java.util.List;

public class PlotTrackersTablesaw {

    private PlotTrackersTablesaw() {
        throw new AssertionError("No PlotTrackersTablesaw!");
    }

    @SneakyThrows
    public static void showAllMetrics(List<Float> epsilons,
                                      List<Float> rewards,
                                      List<Float> losses,
                                      String algorithmName) {

        if (epsilons.isEmpty()) return;

        var table = Table.create("Training Metrics");
        table.addColumns(
                DoubleColumn.create("Episode", range(0, epsilons.size())),
                DoubleColumn.create("Epsilon", toDoubleArray(epsilons)),
                DoubleColumn.create("Reward", toDoubleArray(rewards)),
                DoubleColumn.create("Loss", toDoubleArray(losses))
        );

        var parent = new File("./testoutput/%s/".formatted(algorithmName));
        var template = "graphic_%s_%d.html";
        Files.createDirectories(parent.toPath());
        long timestamp = System.nanoTime();
        Plot.show(LinePlot.create("Epsilon over Episodes", table, "Episode", "Epsilon"),
                new File(parent, template.formatted("epsilon", timestamp)));
        Plot.show(LinePlot.create("Rewards over Episodes", table, "Episode", "Reward"),
                new File(parent, template.formatted("rewards", timestamp)));
        Plot.show(LinePlot.create("Loss over Episodes", table, "Episode", "Loss"),
                new File(parent, template.formatted("loss", timestamp)));
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
