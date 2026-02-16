package br.com.guialves.rflr.gymnasium4j;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import org.bytedeco.cpython.PyObject;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonRuntime.refDec;
import static org.bytedeco.cpython.global.python.PyObject_GetAttrString;

/**
 * Represent the gymnasium.spaces, it can be Box, Discrete and other.
 * <a href="https://gymnasium.farama.org/api/spaces/fundamental/">...</a>
 */
public enum ActionSpaceType {
    DISCRETE {
        @Override
        public ActionResult get(Number value) {
            return new ActionResult(pyLong(value.longValue()), this);
        }
    },
    BOX {
        @Override
        public ActionResult get(Number value) {
            return new ActionResult(pyDouble(value.doubleValue()), this);
        }
    },
    MULTI_DISCRETE {
        @Override
        public ActionResult convert(PyObject obj) {
            // Returns int array
            throw new UnsupportedOperationException("MULTI_DISCRETE is not supported yet!");
        }
    },
    MULTI_BINARY {
        @Override
        public ActionResult convert(PyObject obj) {
            // Returns binary array
            throw new UnsupportedOperationException("MULTI_BINARY is not supported yet!");
        }

        @Override
        public ActionResult get(Number value) {
            return super.get(value);
        }
    },
    TEXT {
        @Override
        public ActionResult convert(PyObject obj) {
            throw new UnsupportedOperationException("TEXT is not supported yet!");
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
        var pyClassName = PyObject_GetAttrString(pySpaceClass, "__name__");

        String name = str(pyClassName, "pyClassName");

        refDec(pyClassName);
        refDec(pySpaceClass);

        return switch (name) {
            case "Discrete" -> ActionSpaceType.DISCRETE;
            case "Box" -> ActionSpaceType.BOX;
            case "MultiDiscrete" -> ActionSpaceType.MULTI_DISCRETE;
            case "MultiBinary" -> ActionSpaceType.MULTI_BINARY;
            default -> ActionSpaceType.UNKNOWN;
        };
    }

    public ActionResult convert(PyObject obj) {
        return new ActionResult(obj, this);
    }

    public ActionResult get(Number value) {
        throw new UnsupportedOperationException("get for \"%s\" is not supported yet!".formatted(this.name()));
    }

    @RequiredArgsConstructor
    @Accessors(fluent = true)
    public static class ActionResult implements AutoCloseable {

        // I don't want to expose this outside of package, to avoid crashes
        final PyObject pyObj;
        @Getter
        final ActionSpaceType spaceType;

        private boolean closed = false;

        @Override
        public void close() {
            if (closed) {
                throw new IllegalStateException("ActionResult already closed!");
            }
            refDec(pyObj);
            closed = true;
        }

        public boolean isClosed() {
            return closed;
        }
    }
}
