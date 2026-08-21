package br.com.guialves.rflr.carla.test;

import org.carla.javacpp.api.Actor;
import org.carla.javacpp.api.Blueprint;
import org.carla.javacpp.api.Client;
import org.carla.javacpp.api.Location;
import org.carla.javacpp.api.Rotation;
import org.carla.javacpp.api.Transform;
import org.carla.javacpp.api.World;

import java.time.Duration;
import java.util.List;

public final class CarlaJavaSmokeTest {
    private CarlaJavaSmokeTest() {
        throw new IllegalStateException("No CarlaJavaSmokeTest");
    }

    static void main(String[] args) {
        var host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 2000;

        try (var client = new Client(host, port)) {
            client.setTimeout(Duration.ofSeconds(10));

            try (var world = client.getWorld()) {
                System.out.println("Connected to CARLA at " + host + ":" + port);
                System.out.println("Map: " + world.getMapName());

                try (var actors = world.getActors()) {
                    System.out.println("Actors before spawn: " + actors.size());
                }

                var spawned = spawnVehicle(world);
                if (spawned == null) {
                    System.out.println("No vehicle blueprint found; connection test completed.");
                    return;
                }

                try (spawned) {
                    System.out.println("Spawned actor id=" + spawned.getId()
                        + ", type=" + spawned.getTypeId());
                    System.out.println("Transform: " + spawned.getTransform());
                    System.out.println("Destroyed: " + spawned.destroy());
                }
            }
        }
    }

    private static Actor spawnVehicle(World world) {
        try (var blueprints = world.getBlueprintLibrary()) {
            List<Blueprint> vehicles = blueprints.filter("vehicle.*");
            if (vehicles.isEmpty()) {
                return null;
            }

            var blueprint = vehicles.get(0).setAttribute("role_name", "java-smoke-test");
            var spawnTransform = new Transform(
                new Location(0.0, 0.0, 1.0),
                new Rotation(0.0, 0.0, 0.0));

            return world.spawnActor(blueprint, spawnTransform);
        }
    }
}
