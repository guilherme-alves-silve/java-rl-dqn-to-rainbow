package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.python.PythonDataStructures;
import br.com.guialves.rflr.python.numpy.NumpyByteBuffer;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.bytedeco.cpython.PyObject;

import java.util.Arrays;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonDataStructures.*;
import static br.com.guialves.rflr.python.PythonTypeChecks.*;
import static org.bytedeco.cpython.global.python.*;

/**
 * Represent the gymnasium.spaces, it can be Box, Discrete and other.
 * <a href="https://gymnasium.farama.org/api/spaces/fundamental/">...</a>
 */
public enum ActionSpaceType {
    DISCRETE {
        @Override
        public ActionResult get(long value) {
            return new ActionResult(pyLong(value), this);
        }

        @Override
        public ActionResult get(int value) {
            return new ActionResult(pyLong(value), this);
        }
    },
    BOX {
        @Override
        public ActionResult get(double value) {
            return new ActionResult(pyDouble(value), this);
        }

        @Override
        public ActionResult get(double[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }

        @Override
        public ActionResult get(float[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }
    },
    MULTI_DISCRETE {
        @Override
        public ActionResult get(int[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }

        @Override
        public ActionResult get(long[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }
    },
    MULTI_BINARY {
        @Override
        public ActionResult get(boolean[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }

        /**
         * accept ints and convert to binary
         */
        @Override
        public ActionResult get(int[] values) {
            return new ActionResult(PythonDataStructures.pyList(values), this);
        }
    },
    TEXT {
        @Override
        public ActionResult get(String value) {
            return new ActionResult(pyStr(value), this);
        }
    },
    UNKNOWN {
        @Override
        public ActionResult convert(PyObject obj) {
            throw new UnsupportedOperationException("UNKNOWN Unsupported PyObject type!");
        }
    };

    static ActionSpaceType detectActionSpaceType(PyObject pyActionSpace) {
        var pySpaceClass = attr(pyActionSpace, "__class__");
        var pyClassName = attr(pySpaceClass, "__name__");

        String name = toStr(pyClassName, "pyClassName");

        decRef(pyClassName);
        decRef(pySpaceClass);

        return switch (name) {
            case "Discrete" -> ActionSpaceType.DISCRETE;
            case "Box" -> ActionSpaceType.BOX;
            case "MultiDiscrete" -> ActionSpaceType.MULTI_DISCRETE;
            case "MultiBinary" -> ActionSpaceType.MULTI_BINARY;
            case "Text" -> ActionSpaceType.TEXT;
            default -> ActionSpaceType.UNKNOWN;
        };
    }

    public ActionResult convert(PyObject obj) {
        return new ActionResult(obj, this);
    }

    public ActionResult get(int value) {
        throw new UnsupportedOperationException("get(int) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(long value) {
        throw new UnsupportedOperationException("get(long) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(double value) {
        throw new UnsupportedOperationException("get(double) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(float value) {
        throw new UnsupportedOperationException("get(float) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(String value) {
        throw new UnsupportedOperationException("get(String) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(int[] values) {
        throw new UnsupportedOperationException("get(int[]) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(long[] values) {
        throw new UnsupportedOperationException("get(long[]) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(double[] values) {
        throw new UnsupportedOperationException("get(double[]) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(float[] values) {
        throw new UnsupportedOperationException("get(float[]) for \"%s\" is not supported!".formatted(this.name()));
    }

    public ActionResult get(boolean[] values) {
        throw new UnsupportedOperationException("get(boolean[]) for \"%s\" is not supported!".formatted(this.name()));
    }

    @RequiredArgsConstructor
    @Accessors(fluent = true)
    public static class ActionResult implements AutoCloseable {

        // I don't want to expose this, to avoid crashes
        final PyObject pyObj;
        @Getter
        final ActionSpaceType spaceType;

        private boolean closed = false;

        @Override
        public void close() {
            if (closed) {
                throw new IllegalStateException("ActionResult already closed!");
            }

            refDecSafe(pyObj);
            closed = true;
        }

        public boolean isClosed() {
            return closed;
        }

        /**
         * Check if this object is still valid (has references)
         */
        public boolean isValid() {
            return !closed && pyObj != null && !pyObj.isNull() && refCount(pyObj) > 0;
        }

        /**
         * Extract the value from the PyObject based on the space type.
         * Returns the appropriate Java type:
         * - DISCRETE: Long
         * - BOX (scalar): Double
         * - BOX (array): double[]
         * - MULTI_DISCRETE: int[] or long[]
         * - MULTI_BINARY: boolean[]
         * - TEXT: String
         *
         * @return the extracted value
         * @param <T> the expected return type
         * @throws IllegalStateException if the ActionResult is closed
         * @throws UnsupportedOperationException if extraction is not supported for the space type
         */
        @SuppressWarnings("unchecked")
        public <T> T value() {
            if (closed) {
                throw new IllegalStateException("Cannot access value of closed ActionResult!");
            }
            if (isPyNull(pyObj)) {
                throw new IllegalStateException("PyObject is null or invalid!");
            }

            return (T) switch (spaceType) {
                case DISCRETE -> toLong(pyObj);
                case BOX -> extractBoxValue(pyObj);
                case MULTI_DISCRETE -> extractMultiDiscreteValue(pyObj);
                case MULTI_BINARY -> toBoolArray(pyObj);
                case TEXT -> toStr(pyObj);
                case UNKNOWN -> throw new UnsupportedOperationException(
                        "Cannot extract value from UNKNOWN space type"
                );
            };
        }

        /**
         * Extract value as a specific type with type checking.
         *
         * @param clazz the expected class type
         * @param <T> the type parameter
         * @return the value cast to the expected type
         * @throws IllegalStateException if closed
         * @throws ClassCastException if the value is not of the expected type
         */
        public <T> T valueAs(Class<T> clazz) {
            Object value = value();
            if (value == null) {
                return null;
            }
            if (!clazz.isInstance(value)) {
                throw new ClassCastException(
                        "Expected %s but got %s".formatted(clazz.getName(), value.getClass().getName())
                );
            }
            return clazz.cast(value);
        }

        /**
         * Extract value for BOX space (can be scalar or array)
         */
        private Object extractBoxValue(PyObject obj) {
            if (isSequence(obj)) {
                return NumpyByteBuffer.toDoubleArray(obj);
            }

            return PyFloat_AsDouble(obj);
        }

        /**
         * Extract value for MULTI_DISCRETE space
         */
        private Object extractMultiDiscreteValue(PyObject obj) {
            if (!isList(obj)) {
                throw new IllegalStateException("Expected list for MULTI_DISCRETE");
            }

            long size = PyList_Size(obj);

            if (size > 0) {
                PyObject first = PyList_GetItem(obj, 0);
                long value = PyLong_AsLong(first);

                if (value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
                    return toIntArray(obj);
                }
            }

            return toLongArray(obj);
        }

        /**
         * Get a string representation of the value for debugging.
         */
        public String valueToString() {
            if (closed) {
                return "[closed]";
            }

            try {
                Object value = value();
                return switch (value) {
                    case null -> "null";
                    case int[] arr -> Arrays.toString(arr);
                    case long[] arr -> Arrays.toString(arr);
                    case double[] arr -> Arrays.toString(arr);
                    case float[] arr -> Arrays.toString(arr);
                    case boolean[] arr -> Arrays.toString(arr);
                    default -> value.toString();
                };

            } catch (Exception e) {
                return "[error: " + e.getMessage() + "]";
            }
        }

        @Override
        public String toString() {
            return "ActionResult{" +
                    "spaceType=" + spaceType +
                    ", value=" + valueToString() +
                    ", closed=" + closed +
                    ", valid=" + isValid() +
                    '}';
        }
    }
}
