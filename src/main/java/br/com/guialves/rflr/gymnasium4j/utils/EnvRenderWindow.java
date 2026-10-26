package br.com.guialves.rflr.gymnasium4j.utils;

import lombok.SneakyThrows;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.Objects.requireNonNull;

public class EnvRenderWindow implements AutoCloseable {

    private final int numberOfEnvs;
    private JFrame frame;
    private JPanel panel;
    private List<JLabel> labels;
    private boolean initialized = false;

    public EnvRenderWindow() {
        this(1);
    }

    public EnvRenderWindow(int numberOfEnvs) {
        this(numberOfEnvs, "Gymnasium Render");
    }

    public EnvRenderWindow(int numberOfEnvs, String windowName) {
        if (numberOfEnvs <= 0) throw new IllegalArgumentException("numberOfEnvs must be greater than or equal to 1!");
        if (requireNonNull(windowName).isBlank()) throw new IllegalArgumentException("windowName must not be blank!");
        this.numberOfEnvs = numberOfEnvs;
        this.frame = new JFrame(windowName);
        this.panel = new JPanel();
        this.labels = IntStream.range(0, numberOfEnvs).mapToObj(_ -> {
            var label = new JLabel();
            panel.add(label);
            return label;
        }).toList();
        this.frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

    public void display(BufferedImage image) {
        display(List.of(image));
    }

    public void display(List<BufferedImage> images) {
        if (images.size() != numberOfEnvs) throw new IllegalArgumentException("images must be equal to numberOfEnvs = " + numberOfEnvs);
        SwingUtilities.invokeLater(() -> {
            for (int i = 0; i < images.size(); ++i) {
                var image = images.get(i);
                var label = labels.get(i);
                var imgIcon = new ImageIcon(image);
                label.setIcon(imgIcon);
            }

            if (!initialized) {
                frame.add(panel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                initialized = true;
            }

            labels.forEach(label -> {
                label.revalidate();
                label.repaint();
            });
        });
    }

    private void closeDisplay() {
        SwingUtilities.invokeLater(() -> {
            if (frame != null) {
                frame.dispose();
                frame = null;
                panel = null;
                labels = null;
                initialized = false;
            }
        });
    }

    public int numberOfEnvs() {
        return numberOfEnvs;
    }

    @Override
    public void close() {
        closeDisplay();
    }

    @SneakyThrows
    public void waitRender() {
        Thread.sleep(1000/60);
    }

    public void displayAndWait(BufferedImage image) {
        this.display(image);
        this.waitRender();
    }

    public void displayAndWait(List<BufferedImage> images) {
        this.display(images);
        this.waitRender();
    }
}
