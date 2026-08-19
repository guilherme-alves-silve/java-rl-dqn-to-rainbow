/** Based on the Python example: manual_control.py */
package br.com.guialves.rflr.carla.test;

import org.carla.javacpp.api.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.time.Duration;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public final class CarlaCameraViewer {

    private static final boolean GOOD_RES = false;

    private CarlaCameraViewer() {
        throw new IllegalStateException("No CarlaCameraViewer");
    }

    static void main(String[] args) throws InterruptedException {
        var host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 2000;

        int width = GOOD_RES? 1280 : 820;
        int height = GOOD_RES? 960 : 640;
        float fov = 90.0f;

        var panel = new ImagePanel();
        var running = new AtomicBoolean(true);
        var cameraMode = new AtomicReference<>(CameraMode.THIRD_PERSON);
        var keyboard = new KeyboardControl();
        SwingUtilities.invokeLater(() -> {
            var frame = new JFrame("CARLA RGB Camera");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.addWindowListener(new java.awt.event.WindowAdapter() {
                @Override
                public void windowClosing(java.awt.event.WindowEvent event) {
                    running.set(false);
                }
            });
            frame.setSize(width, height);
            frame.setContentPane(panel);
            frame.addKeyListener(keyboard);
            keyboard.setCameraToggle(() -> {
                var next = cameraMode.updateAndGet(CameraMode::next);
                IO.println("Camera mode: " + next.label());
            });
            frame.setVisible(true);
            frame.requestFocusInWindow();
        });

        try (var client = new Client(host, port)) {
            client.setTimeout(Duration.ofSeconds(10));

            try (var world = client.getWorld();
                 var blueprints = world.getBlueprintLibrary()) {
                var vehicleBlueprint = first(blueprints.filter("vehicle.*"), "vehicle.*");
                var cameraBlueprintMarker = first(blueprints.filter("sensor.camera.rgb"), "sensor.camera.rgb");
                IO.println("Using camera blueprint: " + cameraBlueprintMarker.getId());

                try (var vehicle = spawnVehicle(world, vehicleBlueprint.setAttribute("role_name", "java-camera-test"));
                     var thirdPersonCamera = world.spawnRgbCamera(
                         vehicle,
                         new Transform(new Location(-5.5, 0.0, 2.8), new Rotation(-15.0, 0.0, 0.0)),
                             (int) (width * 0.95f),
                             (int) (height * 0.95f),
                         fov);
                     var firstPersonCamera = world.spawnRgbCamera(
                         vehicle,
                         new Transform(new Location(1.6, 0.0, 1.7), new Rotation(0.0, 0.0, 0.0)),
                             (int) (width * 0.95f),
                             (int) (height * 0.95f),
                         fov)) {

                    IO.println("Vehicle id: " + vehicle.getId());
                    IO.println("Third person camera id: " + thirdPersonCamera.getId());
                    IO.println("First person camera id: " + firstPersonCamera.getId());
                    IO.println("Controls: W/UP accelerate, S/DOWN brake, A/LEFT steer left, D/RIGHT steer right, SPACE handbrake, R reverse, C camera");

                    thirdPersonCamera.listen(image -> {
                        if (cameraMode.get() == CameraMode.THIRD_PERSON) {
                            SwingUtilities.invokeLater(() -> panel.setImage(image.toBufferedImage()));
                        }
                    });
                    firstPersonCamera.listen(image -> {
                        if (cameraMode.get() == CameraMode.FIRST_PERSON) {
                            SwingUtilities.invokeLater(() -> panel.setImage(image.toBufferedImage()));
                        }
                    });

                    while (running.get()) {
                        var control = keyboard.currentControl();
                        vehicle.applyControl(control);
                        Thread.sleep(25);
                    }

                    thirdPersonCamera.stop();
                    firstPersonCamera.stop();
                    thirdPersonCamera.destroy();
                    firstPersonCamera.destroy();
                    vehicle.destroy();
                }
            }
        }
    }

    private static Blueprint first(List<Blueprint> blueprints, String pattern) {
        if (blueprints.isEmpty()) {
            throw new IllegalStateException("No blueprint found for " + pattern);
        }
        return blueprints.getFirst();
    }

    private static Vehicle spawnVehicle(World world, Blueprint blueprint) {
        List<Transform> spawnPoints = world.getSpawnPoints();
        if (spawnPoints.isEmpty()) {
            throw new IllegalStateException("Current map has no recommended spawn points");
        }

        var random = new Random();
        int start = random.nextInt(spawnPoints.size());
        for (int i = 0; i < spawnPoints.size(); i++) {
            var spawnPoint = spawnPoints.get((start + i) % spawnPoints.size());
            var vehicle = world.trySpawnVehicle(blueprint, spawnPoint);
            if (vehicle != null) {
                IO.println("Spawn point: " + spawnPoint);
                return vehicle;
            }
        }

        throw new IllegalStateException("Could not spawn vehicle at any recommended spawn point");
    }

    private enum CameraMode {
        THIRD_PERSON("third person"),
        FIRST_PERSON("first person");

        private final String label;

        CameraMode(String label) {
            this.label = label;
        }

        String label() {
            return label;
        }

        CameraMode next() {
            return this == THIRD_PERSON ? FIRST_PERSON : THIRD_PERSON;
        }
    }

    private static final class ImagePanel extends JPanel {
        private BufferedImage image;

        void setImage(BufferedImage image) {
            this.image = image;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);
            if (image != null) {
                graphics.drawImage(image, 0, 0, getWidth(), getHeight(), null);
            }
        }
    }

    private static final class KeyboardControl extends KeyAdapter {
        private final Set<Integer> pressed = new HashSet<>();
        private boolean reverse;
        private Runnable cameraToggle;

        synchronized void setCameraToggle(Runnable cameraToggle) {
            this.cameraToggle = cameraToggle;
        }

        @Override
        public synchronized void keyPressed(KeyEvent event) {
            if (event.getKeyCode() == KeyEvent.VK_R) {
                reverse = !reverse;
                IO.println("Reverse: " + reverse);
            }
            if (event.getKeyCode() == KeyEvent.VK_C && !pressed.contains(KeyEvent.VK_C) && cameraToggle != null) {
                cameraToggle.run();
            }
            pressed.add(event.getKeyCode());
        }

        @Override
        public synchronized void keyReleased(KeyEvent event) {
            pressed.remove(event.getKeyCode());
        }

        synchronized VehicleControl currentControl() {
            boolean accelerate = pressed.contains(KeyEvent.VK_W) || pressed.contains(KeyEvent.VK_UP);
            boolean brakeKey = pressed.contains(KeyEvent.VK_S) || pressed.contains(KeyEvent.VK_DOWN);
            boolean left = pressed.contains(KeyEvent.VK_A) || pressed.contains(KeyEvent.VK_LEFT);
            boolean right = pressed.contains(KeyEvent.VK_D) || pressed.contains(KeyEvent.VK_RIGHT);
            boolean handBrake = pressed.contains(KeyEvent.VK_SPACE);

            float throttle = accelerate ? 0.65f : 0.0f;
            float brake = brakeKey ? 0.75f : 0.0f;
            float steer = 0.0f;
            if (left && !right) {
                steer = -0.45f;
            } else if (right && !left) {
                steer = 0.45f;
            }

            return new VehicleControl()
                .throttle(throttle)
                .steer(steer)
                .brake(brake)
                .handBrake(handBrake)
                .reverse(reverse);
        }
    }
}
