package br.com.guialves.rflr.playground;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.Cleanup;

import java.util.ArrayList;
import java.util.List;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;

public class PlaygroundMemoryManagement {

    static void main() {
        var released = new ArrayList<NDArray>();
        {
            @Cleanup var parent = setName(NDManager.newBaseManager(), "parent");
            {
                var p1 = parent.create(1);
                var p2 = parent.create(2);
                released.addAll(List.of(p1, p2));
                @Cleanup var inner1 = subMgr(parent, "inner-1");
                var i1 = inner1.create(1);
                var i2 = inner1.create(2);
                released.addAll(List.of(i1, i2));
                {
                    @Cleanup var inner2 = subMgr(inner1, "inner-2");
                    var ii1 = inner2.create(1);
                    @Cleanup var inner3 = subMgr(inner1, "inner-3");
                    var ii2 = inner3.create(2);
                    released.addAll(List.of(ii1, ii2));
                    IO.println("**** parent with inner1, inner2 and inner3 ****");
                    debugDump(parent);
                }
                IO.println("**** parent with inner1 (without inner2) ****");
                debugDump(parent);
            }
            IO.println("**** only parent ****");
            debugDump(parent);
        }
        var allReleased = released.stream().allMatch(NDArray::isReleased);
        IO.println("All released: " + allReleased);
    }
}
