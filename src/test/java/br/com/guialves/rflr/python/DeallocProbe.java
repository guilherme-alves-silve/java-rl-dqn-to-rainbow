package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.Py_buffer;
import org.bytedeco.javacpp.Pointer;

import java.lang.reflect.Field;

import static br.com.guialves.rflr.python.PythonRuntime.*;

/**
 * Direct (runtime) proof that {@code close()} and {@code retainReference()} are no-ops
 * for {@code PyObject}, because there is no deallocator.
 *
 * <p>Proof chain:
 * <ol>
 *   <li>{@code PyObject} is {@code @Opaque} and does not declare a deallocator -> the field
 *       {@code Pointer.deallocator} remains {@code null} (read via reflection here).</li>
 *   <li>{@code Pointer.close()} -> {@code releaseReference()} only acts if
 *       {@code deallocator instanceof DeallocatorReference} -> {@code null} -> nothing happens.</li>
 *   <li>{@code Pointer.retainReference()} only acts if {@code deallocator instanceof
 *       ReferenceCounter} -> {@code null} -> nothing happens.</li>
 * </ol>
 */
public class DeallocProbe {

    public static void main(String[] args) throws Exception {
        initPython();

        Field field = Pointer.class.getDeclaredField("deallocator");   // Pointer.java:606
        field.setAccessible(true);

        exec("class _Foo: pass\n_f1 = _Foo()\n_f2 = _f1");   // 2 global references
        PyObject py = eval("_f1");                          // +1 reference -> 3

        IO.println("=== Pointer.deallocator field (line 606) ===");
        IO.println("PyObject    (@Opaque, without deallocate()) -> " + field.get(py));
        var buffer = new Py_buffer();                         // class not opaque, with allocate()
        IO.println("Py_buffer  (not @Opaque, with allocate()) -> " + field.get(buffer));

        IO.println();
        IO.println("=== effect of close() on CPython refcount ===");
        long before = refCount(py);
        py.close();
        long after = refCount(py);
        IO.println("sys.getrefcount before close()  = " + before);
        IO.println("sys.getrefcount after close()   = " + after);
        IO.println("did close() change the refcount? " + (before != after));
        IO.println("isNull() after close() = " + py.isNull() + "  (pointer remains valid)");
        IO.println("address after close() = " + py.address());

        IO.println();
        IO.println("=== effect of retainReference() ===");
        long b2 = refCount(py);
        py.retainReference();
        long a2 = refCount(py);
        IO.println("refcount before = " + b2 + ", after = " + a2
                + "  -> retainReference() " + (b2 == a2 ? "did not change the CPython refcount (no-op)" : "changed it"));

        IO.println();
        IO.println("=== contrast: Py_buffer HAS a deallocator -> close() REALLY frees ===");
        IO.println("Py_buffer deallocator = " + field.get(buffer));
        buffer.close();
        IO.println("Py_buffer address after close() = " + buffer.address()
                + "  (0 = actually deallocated)");
    }
}
