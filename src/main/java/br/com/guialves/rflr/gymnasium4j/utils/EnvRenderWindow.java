package br.com.guialves.rflr.gymnasium4j.utils;

import lombok.SneakyThrows;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import java.awt.image.BufferedImage;

public class EnvRenderWindow implements AutoCloseable {

    private JFrame frame;
    private JLabel label;
    private boolean initialized = false;

    public EnvRenderWindow() {
        this("Gymnasium Render");
    }

    public EnvRenderWindow(String windowName) {
        this.frame = new JFrame(windowName);
        this.label = new JLabel();
        this.frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

    public void display(BufferedImage image) {
        SwingUtilities.invokeLater(() -> {
            var imgIcon = new ImageIcon(image);
            if (!initialized) {
                frame.add(label);
                label.setIcon(imgIcon);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                initialized = true;
            } else {
                label.setIcon(imgIcon);
                label.revalidate();
                label.repaint();
            }
        });
    }

    private void closeDisplay() {
        SwingUtilities.invokeLater(() -> {
            if (frame != null) {
                frame.dispose();
                frame = null;
                label = null;
                initialized = false;
            }
        });
    }

    @Override
    public void close() {
        closeDisplay();
    }

    @SneakyThrows
    public void waitRender() {
        Thread.sleep(1000/60);
    }
}
