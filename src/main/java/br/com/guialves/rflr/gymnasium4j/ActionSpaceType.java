package br.com.guialves.rflr.gymnasium4j;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.bytedeco.cpython.PyObject;

import static br.com.guialves.rflr.python.PythonRuntime.*;

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
            return new ActionResult(pyListDoubles(values), this);
        }

        @Override
        public ActionResult get(float[] values) {
            return new ActionResult(pyListFloats(values), this);
        }
    },
    MULTI_DISCRETE {
        @Override
        public ActionResult get(int[] values) {
            return new ActionResult(pyListInts(values), this);
        }

        @Override
        public ActionResult get(long[] values) {
            return new ActionResult(pyListLongs(values), this);
        }
    },
    MULTI_BINARY {
        @Override
        public ActionResult get(boolean[] values) {
            return new ActionResult(pyListBools(values), this);
        }

        /**
         * accept ints and convert to binary
         */
        @Override
        public ActionResult get(int[] values) {
            return new ActionResult(pyListInts(values), this);
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

        String name = str(pyClassName, "pyClassName");

        refDec(pyClassName);
        refDec(pySpaceClass);

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

        public Object value() {
            if (closed) {
                throw new IllegalStateException("Cannot get value from closed ActionResult");
            }
            return null;
        }

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
    }
}
