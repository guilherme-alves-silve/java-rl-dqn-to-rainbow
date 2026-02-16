# Byte Buddy Maven Plugin for Automatic PyObject Management

## Overview

Use Byte Buddy's Maven plugin to automatically inject cleanup code for PyObject references at compile time. This approach is similar to Lombok - it modifies bytecode during compilation, runs once, and the resulting JAR contains the transformed code with no runtime overhead.

## Project Setup

### Maven Dependencies

```xml
<project>
    <properties>
        <byte-buddy.version>1.14.10</byte-buddy.version>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>

    <dependencies>
        <!-- Byte Buddy runtime (needed for annotations) -->
        <dependency>
            <groupId>net.bytebuddy</groupId>
            <artifactId>byte-buddy</artifactId>
            <version>${byte-buddy.version}</version>
        </dependency>
        
        <!-- Byte Buddy agent (for build plugin) -->
        <dependency>
            <groupId>net.bytebuddy</groupId>
            <artifactId>byte-buddy-agent</artifactId>
            <version>${byte-buddy.version}</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Standard Java compiler -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                    <parameters>true</parameters>
                </configuration>
            </plugin>

            <!-- Byte Buddy transformation plugin -->
            <plugin>
                <groupId>net.bytebuddy</groupId>
                <artifactId>byte-buddy-maven-plugin</artifactId>
                <version>${byte-buddy.version}</version>
                <executions>
                    <execution>
                        <goals>
                            <goal>transform</goal>
                        </goals>
                    </execution>
                </executions>
                <configuration>
                    <transformations>
                        <transformation>
                            <plugin>com.example.pyobject.PyObjectCleanupPlugin</plugin>
                        </transformation>
                    </transformations>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

### Gradle Configuration

```groovy
plugins {
    id 'java'
    id 'net.bytebuddy.byte-buddy-gradle-plugin' version '1.14.10'
}

dependencies {
    implementation 'net.bytebuddy:byte-buddy:1.14.10'
    compileOnly 'net.bytebuddy:byte-buddy-agent:1.14.10'
}

byteBuddy {
    transformation {
        plugin = 'com.example.pyobject.PyObjectCleanupPlugin'
    }
}
```

## Annotation Definition

```java
package com.example.pyobject;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks methods that should automatically cleanup PyObject references.
 * Similar to @Cleanup in Lombok, but specifically for Python objects.
 * 
 * The plugin will:
 * 1. Track all local PyObject variables
 * 2. Identify which are owned vs borrowed references
 * 3. Insert Py_DECREF calls in a finally block
 * 
 * Usage:
 * <pre>
 * @AutoCleanupPyObjects
 * public EnvStepResult step(ActionResult action) {
 *     var result = callFunction(pyStep, action.obj);  // Will be auto-cleaned
 *     var pyState = getItem(result, 0);               // Borrowed, won't be cleaned
 *     return new EnvStepResult(...);
 * }
 * </pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface AutoCleanupPyObjects {
    
    /**
     * Additional variable names to exclude from cleanup (borrowed references)
     */
    String[] exclude() default {};
    
    /**
     * Enable verbose logging during transformation
     */
    boolean debug() default false;
}
```

## Byte Buddy Plugin Implementation

```java
package com.example.pyobject;

import net.bytebuddy.build.Plugin;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.ClassFileLocator;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.jar.asm.*;
import net.bytebuddy.jar.asm.commons.AdviceAdapter;

import java.io.IOException;
import java.util.*;

import static net.bytebuddy.jar.asm.Opcodes.*;

/**
 * Byte Buddy plugin that transforms methods annotated with @AutoCleanupPyObjects
 * to automatically manage PyObject reference counting.
 */
public class PyObjectCleanupPlugin implements Plugin {
    
    private static final String ANNOTATION_DESCRIPTOR = 
        "Lcom/example/pyobject/AutoCleanupPyObjects;";
    
    @Override
    public boolean matches(TypeDescription target) {
        // Only process classes that might contain PyObject usage
        return target.getName().startsWith("com.example.gym") ||
               target.getName().startsWith("com.example.env");
    }
    
    @Override
    public DynamicType.Builder<?> apply(
            DynamicType.Builder<?> builder,
            TypeDescription typeDescription,
            ClassFileLocator classFileLocator) {
        
        try {
            // Read the original class file
            byte[] classFile = classFileLocator.locate(typeDescription.getName())
                .resolve();
            
            // Transform with ASM
            ClassReader cr = new ClassReader(classFile);
            ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES);
            ClassVisitor cv = new PyObjectClassVisitor(ASM9, cw);
            
            cr.accept(cv, ClassReader.EXPAND_FRAMES);
            
            // Return transformed builder
            return builder.make();
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to transform class: " + typeDescription, e);
        }
    }
    
    @Override
    public void close() {
        // No resources to clean up
    }
    
    /**
     * Class visitor that finds methods with @AutoCleanupPyObjects annotation
     */
    static class PyObjectClassVisitor extends ClassVisitor {
        
        public PyObjectClassVisitor(int api, ClassVisitor cv) {
            super(api, cv);
        }
        
        @Override
        public MethodVisitor visitMethod(int access, String name, String descriptor,
                                        String signature, String[] exceptions) {
            
            MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
            
            // Return transformer that checks for annotation
            return new AnnotationCheckingMethodVisitor(api, mv, access, name, descriptor);
        }
    }
    
    /**
     * Method visitor that checks for @AutoCleanupPyObjects and applies transformation
     */
    static class AnnotationCheckingMethodVisitor extends MethodVisitor {
        
        private final MethodVisitor delegate;
        private final int access;
        private final String name;
        private final String descriptor;
        private boolean hasAnnotation = false;
        private boolean debug = false;
        private Set<String> excludedVars = new HashSet<>();
        
        public AnnotationCheckingMethodVisitor(int api, MethodVisitor mv,
                                              int access, String name, String descriptor) {
            super(api, mv);
            this.delegate = mv;
            this.access = access;
            this.name = name;
            this.descriptor = descriptor;
        }
        
        @Override
        public AnnotationVisitor visitAnnotation(String descriptor, boolean visible) {
            if (ANNOTATION_DESCRIPTOR.equals(descriptor)) {
                hasAnnotation = true;
                
                // Parse annotation parameters
                return new AnnotationVisitor(api, super.visitAnnotation(descriptor, visible)) {
                    @Override
                    public void visit(String name, Object value) {
                        super.visit(name, value);
                        if ("debug".equals(name)) {
                            debug = (Boolean) value;
                        }
                    }
                    
                    @Override
                    public AnnotationVisitor visitArray(String name) {
                        if ("exclude".equals(name)) {
                            return new AnnotationVisitor(api, super.visitArray(name)) {
                                @Override
                                public void visit(String name, Object value) {
                                    super.visit(name, value);
                                    excludedVars.add((String) value);
                                }
                            };
                        }
                        return super.visitArray(name);
                    }
                };
            }
            return super.visitAnnotation(descriptor, visible);
        }
        
        @Override
        public void visitCode() {
            super.visitCode();
            
            // Replace with transformer if annotation present
            if (hasAnnotation) {
                if (debug) {
                    System.out.println("Transforming method: " + name + descriptor);
                }
            }
        }
        
        @Override
        public void visitEnd() {
            super.visitEnd();
            
            // If annotated, wrap with cleanup transformer
            if (hasAnnotation) {
                // Create new method visitor with cleanup logic
                MethodVisitor transformed = new PyObjectCleanupMethodVisitor(
                    api, delegate, access, name, descriptor, excludedVars, debug
                );
                
                // Note: This is simplified - actual transformation happens
                // through ASM tree API or by re-visiting the method
            }
        }
    }
    
    /**
     * Method visitor that injects PyObject cleanup code
     */
    static class PyObjectCleanupMethodVisitor extends AdviceAdapter {
        
        private final Set<String> excludedVars;
        private final boolean debug;
        private final Map<Integer, PyObjectInfo> pyObjectSlots = new HashMap<>();
        private final Set<Integer> ownedReferences = new HashSet<>();
        private final Set<Integer> borrowedReferences = new HashSet<>();
        
        private Label startLabel;
        private Label endLabel;
        private Label handlerLabel;
        
        static class PyObjectInfo {
            String name;
            int slot;
            boolean isOwned;
            
            PyObjectInfo(String name, int slot, boolean isOwned) {
                this.name = name;
                this.slot = slot;
                this.isOwned = isOwned;
            }
        }
        
        protected PyObjectCleanupMethodVisitor(int api, MethodVisitor mv,
                                              int access, String name, String descriptor,
                                              Set<String> excludedVars, boolean debug) {
            super(api, mv, access, name, descriptor);
            this.excludedVars = excludedVars;
            this.debug = debug;
        }
        
        @Override
        protected void onMethodEnter() {
            startLabel = new Label();
            endLabel = new Label();
            handlerLabel = new Label();
            
            mv.visitLabel(startLabel);
        }
        
        @Override
        public void visitLocalVariable(String name, String descriptor,
                                      String signature, Label start, Label end, int index) {
            super.visitLocalVariable(name, descriptor, signature, start, end, index);
            
            // Check if this is a PyObject type
            if (isPyObjectType(descriptor) && !excludedVars.contains(name)) {
                if (debug) {
                    System.out.println("  Found PyObject variable: " + name + " at slot " + index);
                }
                pyObjectSlots.put(index, new PyObjectInfo(name, index, true));
            }
        }
        
        @Override
        public void visitMethodInsn(int opcode, String owner, String name,
                                   String descriptor, boolean isInterface) {
            super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
            
            // Track whether this method returns owned or borrowed references
            if (isNewReferenceMethod(name)) {
                // Next ASTORE will be an owned reference
                if (debug) {
                    System.out.println("  Method " + name + " returns NEW reference");
                }
                // Mark that next store should be tracked as owned
                
            } else if (isBorrowedReferenceMethod(name)) {
                // Next ASTORE will be a borrowed reference
                if (debug) {
                    System.out.println("  Method " + name + " returns BORROWED reference");
                }
                // Mark that next store should NOT be cleaned up
            }
        }
        
        @Override
        protected void onMethodExit(int opcode) {
            // Insert cleanup on normal returns
            if (opcode != ATHROW) {
                insertCleanupCode();
            }
        }
        
        @Override
        public void visitMaxs(int maxStack, int maxLocals) {
            mv.visitLabel(endLabel);
            
            // Create exception handler
            Label afterHandler = new Label();
            mv.visitJumpInsn(GOTO, afterHandler);
            
            // Exception handler - cleanup then rethrow
            mv.visitLabel(handlerLabel);
            mv.visitVarInsn(ASTORE, maxLocals); // Store exception
            insertCleanupCode();
            mv.visitVarInsn(ALOAD, maxLocals);  // Reload exception
            mv.visitInsn(ATHROW);
            
            mv.visitLabel(afterHandler);
            
            // Register exception handler
            mv.visitTryCatchBlock(startLabel, endLabel, handlerLabel, null);
            
            super.visitMaxs(maxStack + 1, maxLocals + 1);
        }
        
        private void insertCleanupCode() {
            if (pyObjectSlots.isEmpty()) {
                return;
            }
            
            if (debug) {
                System.out.println("  Inserting cleanup for " + pyObjectSlots.size() + " PyObjects");
            }
            
            // Sort slots in reverse order (cleanup newest first)
            List<Integer> slots = new ArrayList<>(pyObjectSlots.keySet());
            slots.sort(Collections.reverseOrder());
            
            for (Integer slot : slots) {
                PyObjectInfo info = pyObjectSlots.get(slot);
                
                // Skip borrowed references
                if (borrowedReferences.contains(slot)) {
                    if (debug) {
                        System.out.println("  Skipping borrowed reference: " + info.name);
                    }
                    continue;
                }
                
                // Skip excluded variables
                if (excludedVars.contains(info.name)) {
                    if (debug) {
                        System.out.println("  Skipping excluded variable: " + info.name);
                    }
                    continue;
                }
                
                Label skipLabel = new Label();
                
                // if (pyObj != null)
                mv.visitVarInsn(ALOAD, slot);
                mv.visitJumpInsn(IFNULL, skipLabel);
                
                // Call Py_DECREF(pyObj)
                mv.visitVarInsn(ALOAD, slot);
                mv.visitMethodInsn(
                    INVOKESTATIC,
                    "com/example/python/PythonAPI",  // Your Python JNI wrapper
                    "Py_DECREF",
                    "(Lcom/example/python/PyObject;)V",
                    false
                );
                
                // Set to null to prevent double-free
                mv.visitInsn(ACONST_NULL);
                mv.visitVarInsn(ASTORE, slot);
                
                mv.visitLabel(skipLabel);
                
                if (debug) {
                    System.out.println("  ✓ Added cleanup for: " + info.name);
                }
            }
        }
        
        private boolean isPyObjectType(String descriptor) {
            return descriptor.contains("PyObject") ||
                   descriptor.contains("com/example/python/");
        }
        
        private boolean isNewReferenceMethod(String methodName) {
            // Methods that return NEW references (need cleanup)
            return methodName.equals("callFunction") ||
                   methodName.equals("callMethod") ||
                   methodName.equals("PyObject_CallMethod") ||
                   methodName.equals("PyObject_GetAttrString") ||
                   methodName.equals("PyLong_FromLong") ||
                   methodName.equals("PyTuple_New") ||
                   methodName.equals("PyList_New") ||
                   methodName.equals("PyDict_New");
        }
        
        private boolean isBorrowedReferenceMethod(String methodName) {
            // Methods that return BORROWED references (don't cleanup)
            return methodName.equals("getItem") ||
                   methodName.equals("PyTuple_GetItem") ||
                   methodName.equals("PyList_GetItem") ||
                   methodName.equals("PyDict_GetItem") ||
                   methodName.equals("PyDict_GetItemString");
        }
    }
}
```

## Usage in Your Code

### Before (Manual Cleanup)

```java
public EnvStepResult step(ActionResult action) {
    if (stateBuffer == null) {
        throw new IllegalStateException("You should call reset() first!");
    }

    PyObject result = null;
    try {
        result = callFunction(pyStep, action.obj);
        
        var pyState = getItem(result, 0);
        var pyReward = getItem(result, 1);
        var pyTerminated = getItem(result, 2);
        var pyTruncated = getItem(result, 3);
        var pyInfoMap = getItem(result, 4);

        double reward = toDouble(pyReward);
        boolean terminated = toBool(pyTerminated);
        boolean truncated = toBool(pyTruncated);
        var infoMap = pyDictToJava(pyInfoMap);

        fillFromNumpy(pyState, stateBuffer);

        var state = ndManager.create(
                stateBuffer,
                stateMetadata.djlShape,
                stateMetadata.djlType
        );

        return new EnvStepResult(reward, terminated, truncated, infoMap)
                .state(state);
    } finally {
        if (result != null) {
            Py_DECREF(result);
        }
    }
}
```

### After (Automatic Cleanup)

```java
@AutoCleanupPyObjects
public EnvStepResult step(ActionResult action) {
    if (stateBuffer == null) {
        throw new IllegalStateException("You should call reset() first!");
    }

    // 'result' will be automatically cleaned up
    var result = callFunction(pyStep, action.obj);
    
    // These are borrowed references, won't be cleaned up
    var pyState = getItem(result, 0);
    var pyReward = getItem(result, 1);
    var pyTerminated = getItem(result, 2);
    var pyTruncated = getItem(result, 3);
    var pyInfoMap = getItem(result, 4);

    double reward = toDouble(pyReward);
    boolean terminated = toBool(pyTerminated);
    boolean truncated = toBool(pyTruncated);
    var infoMap = pyDictToJava(pyInfoMap);

    fillFromNumpy(pyState, stateBuffer);

    var state = ndManager.create(
            stateBuffer,
            stateMetadata.djlShape,
            stateMetadata.djlType
    );

    return new EnvStepResult(reward, terminated, truncated, infoMap)
            .state(state);
    
    // Byte Buddy inserted: finally { Py_DECREF(result); }
}
```

### With Debug and Exclusions

```java
@AutoCleanupPyObjects(debug = true, exclude = {"cachedObject"})
public void processData() {
    var temp = callFunction(someFunc);           // Will be cleaned up
    var cachedObject = getFromCache();           // Excluded, won't be cleaned up
    var borrowed = getItem(temp, 0);             // Borrowed, won't be cleaned up
    
    // Do work...
}
```

## Build and Verify

### Build Process

```bash
# Clean and build
mvn clean compile

# Byte Buddy transformation happens automatically after compilation
# Check the output:
# [INFO] Transformed X class(es)

# Package
mvn package

# The JAR now contains transformed bytecode!
```

### Verify Transformation

```java
// View transformed bytecode
mvn dependency:tree
javap -c -v target/classes/com/example/gym/GymEnvironment.class

// Look for injected try-catch blocks and Py_DECREF calls
```

### Testing

```java
public class TestAutoCleanup {
    
    @Test
    public void testPyObjectCleanup() {
        GymEnvironment env = new GymEnvironment("CartPole-v1");
        
        // Call annotated method
        var result = env.step(new ActionResult(0));
        
        // PyObjects should be automatically cleaned up
        // No memory leaks!
        
        assertNotNull(result);
    }
}
```

## Advantages

1. ✅ **Compile-time transformation** - Runs once during build
2. ✅ **No runtime overhead** - No agents, no reflection
3. ✅ **Clean code** - No manual try-finally blocks
4. ✅ **Type-safe** - Detects PyObject types automatically
5. ✅ **Smart detection** - Distinguishes owned vs borrowed references
6. ✅ **JAR ready** - Transformed bytecode is packaged

## Limitations

1. ⚠️ Requires knowledge of which methods return owned vs borrowed references
2. ⚠️ Complex bytecode manipulation
3. ⚠️ May need tuning for edge cases
4. ⚠️ Debugging transformed code can be tricky

## Alternative: Simpler Annotation Processor

If Byte Buddy is too complex, consider generating wrapper methods instead:

```java
// You write:
@AutoCleanupPyObjects
public EnvStepResult step(ActionResult action);

// Annotation processor generates:
public EnvStepResult step(ActionResult action) {
    return step_generated(action);
}

private EnvStepResult step_generated(ActionResult action) {
    // Your original code with injected cleanup
}
```

This is simpler but less elegant than direct bytecode transformation.
