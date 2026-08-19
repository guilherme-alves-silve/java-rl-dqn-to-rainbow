/** Based on the Python example: generate_traffic.py */
package br.com.guialves.rflr.carla.test;

import org.carla.javacpp.api.Blueprint;
import org.carla.javacpp.api.Client;
import org.carla.javacpp.api.Transform;
import org.carla.javacpp.api.Vehicle;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class CarlaTrafficExample {
    private CarlaTrafficExample() {
        throw new IllegalStateException("No CarlaTrafficExample");
    }

    static void main(String[] args) throws Exception {
        int count = args.length > 0 ? Integer.parseInt(args[0]) : 120;
        var vehicles = new ArrayList<Vehicle>();

        try (var client = new Client("localhost", 2000)) {
            client.setTimeout(Duration.ofSeconds(10));
            try (var world = client.getWorld();
                 var blueprints = world.getBlueprintLibrary()) {
                List<Blueprint> vehicleBlueprints = blueprints.filter("vehicle.*");
                List<Transform> spawnPoints = world.getSpawnPoints();
                var random = new Random();

                for (int i = 0; i < count; i++) {
                    var blueprint = vehicleBlueprints.get(random.nextInt(vehicleBlueprints.size()));
                    var spawnPoint = spawnPoints.get(random.nextInt(spawnPoints.size()));
                    var vehicle = world.trySpawnVehicle(blueprint, spawnPoint);
                    if (vehicle != null) {
                        vehicle.setAutopilot(true);
                        vehicles.add(vehicle);
                        System.out.println("Spawned vehicle " + vehicle.getId());
                    }
                }

                Thread.sleep(60_000);
            } finally {
                for (var vehicle : vehicles) {
                    vehicle.destroy();
                    vehicle.close();
                }
            }
        }
    }
}
