package br.com.guialves.rflr.dqn.utils;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import java.awt.image.BufferedImage;

public class EnvRenderWindow implements AutoCloseable {

    private JFrame frame;
    private JLabel label;

    public EnvRenderWindow() {
        this("Gymnasium Render");
    }

    public EnvRenderWindow(String windowName) {
        this.frame = new JFrame(windowName);
        this.label = new JLabel();
    }

    public void displayImage(BufferedImage image) {
        SwingUtilities.invokeLater(() -> {
            frame.add(label);
            frame.setLocationRelativeTo(null);

            label.setIcon(new ImageIcon(image));
            frame.pack();

            if (!frame.isVisible()) {
                frame.setVisible(true);
            }
        });
    }

    private void closeDisplay() {
        SwingUtilities.invokeLater(() -> {
            if (frame != null) {
                frame.dispose();
                frame = null;
                label = null;
            }
        });
    }

    @Override
    public void close() {
        closeDisplay();
    }
}
