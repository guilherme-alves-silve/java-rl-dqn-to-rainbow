/** Based on the Python example: dynamic_weather.py */
package br.com.guialves.rflr.carla.test;

import org.carla.javacpp.api.Client;
import org.carla.javacpp.api.WeatherParameters;

import java.time.Duration;

public final class CarlaWeatherExample {
    private CarlaWeatherExample() {
        throw new IllegalStateException("No CarlaWeatherExample");
    }

    static void main() throws Exception {
        try (var client = new Client("localhost", 2000)) {
            client.setTimeout(Duration.ofSeconds(10));
            try (var world = client.getWorld()) {
                var original = world.getWeather();
                try {
                    for (int i = 0; i <= 100; i += 5) {
                        var weather = WeatherParameters.clearNoon()
                            .cloudiness(i)
                            .precipitation(Math.max(0, i - 30))
                            .sunAzimuthAngle(i * 3.6f)
                            .sunAltitudeAngle(60.0f - i * 0.8f);
                        world.setWeather(weather);
                        System.out.println("Weather cloudiness=" + i + " precipitation=" + Math.max(0, i - 30));
                        Thread.sleep(500);
                    }
                } finally {
                    world.setWeather(original);
                }
            }
        }
    }
}
