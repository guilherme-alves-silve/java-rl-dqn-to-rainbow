package br.com.guialves.rflr.utils;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("ModelFileNames: safe names for the saved models")
class ModelFileNamesTest {

    @Nested
    @DisplayName("sanitize() removes the characters that break the file name")
    class Sanitize {

        @Test
        @DisplayName("The path separator of the environment name is replaced")
        void shouldReplacePathSeparator() {
            assertEquals("ALE-Galaxian-v5", ModelFileNames.sanitize("ALE/Galaxian-v5"));
        }

        @Test
        @DisplayName("The windows path separator is replaced")
        void shouldReplaceWindowsPathSeparator() {
            assertEquals("ALE-Galaxian-v5", ModelFileNames.sanitize("ALE\\Galaxian-v5"));
        }

        @Test
        @DisplayName("The characters reserved by windows are replaced")
        void shouldReplaceReservedCharacters() {
            assertEquals("a-b-c-d-e-f-g", ModelFileNames.sanitize("a:b?c*d\"e<f>g"));
        }

        @Test
        @DisplayName("A name without invalid characters is preserved")
        void shouldPreserveValidName() {
            assertEquals("LunarLander-v3", ModelFileNames.sanitize("LunarLander-v3"));
            assertEquals("galaxian_noisy_nets_dqn",
                    ModelFileNames.sanitize("galaxian_noisy_nets_dqn"));
        }

        @Test
        @DisplayName("A blank name falls back to a default")
        void shouldFallbackToDefault() {
            assertEquals("model", ModelFileNames.sanitize(null));
            assertEquals("model", ModelFileNames.sanitize("   "));
        }
    }

    @Nested
    @DisplayName("build() creates a name that is safe to be used as a DJL prefix")
    class Build {

        @Test
        @DisplayName("The name of an ALE environment has no path separator")
        void shouldNotContainPathSeparator() {
            var name = ModelFileNames.build("ALE/Galaxian-v5", "galaxian_noisy_nets_dqn");

            assertFalse(name.contains("/"), "The name must not create a folder: " + name);
            assertFalse(name.contains("\\"), "The name must not create a folder: " + name);
            assertTrue(name.startsWith("ALE-Galaxian-v5_"), "Unexpected name: " + name);
            assertTrue(name.endsWith("_galaxian_noisy_nets_dqn"), "Unexpected name: " + name);
        }

        @Test
        @DisplayName("The name of an environment without a separator is preserved")
        void shouldPreserveFlatEnvironmentName() {
            var name = ModelFileNames.build("LunarLander-v3", "dqn");

            assertTrue(name.startsWith("LunarLander-v3_"), "Unexpected name: " + name);
            assertTrue(name.endsWith("_dqn"), "Unexpected name: " + name);
        }
    }

    @Nested
    @DisplayName("ensureParentDirectory() creates the folder that will hold the model")
    class EnsureParentDirectory {

        @Test
        @DisplayName("The base folder is created for a flat model name")
        void shouldCreateBaseFolder(@TempDir Path tempDir) {
            var target = tempDir.resolve("output_models").resolve("dqn");
            ModelFileNames.ensureParentDirectory(target, "LunarLander-v3_1_dqn");
            assertTrue(Files.isDirectory(target));
        }

        @Test
        @DisplayName("The nested folder is created for a model name with a separator")
        void shouldCreateNestedFolder(@TempDir Path tempDir) {
            var target = tempDir.resolve("output_models").resolve("dqn");
            ModelFileNames.ensureParentDirectory(target, "ALE/LunarLander-v3_1_dqn");
            assertTrue(Files.isDirectory(target.resolve("ALE")));
        }
    }
}
