package br.com.guialves.rflr.utils.json;

import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import com.google.gson.*;
import java.lang.reflect.Type;

public class ManagerNodeSerializer implements JsonSerializer<DJLMemoryManagement.ManagerNode> {

    @Override
    public JsonElement serialize(DJLMemoryManagement.ManagerNode node, Type typeOfSrc, JsonSerializationContext context) {
        var json = new JsonObject();
        json.addProperty("name", node.name());
        json.addProperty("uid", node.uid());
        json.addProperty("open", node.open());
        json.addProperty("parent", node.parent());
        json.addProperty("totalResources", node.totalResources());

        if (node.parent()) {
            json.addProperty("committedMemBytes", node.committedMemBytes());
            json.addProperty("totalMemBytes", node.totalMemBytes());
            json.addProperty("freeMemBytes", node.freeMemBytes());
            json.addProperty("javaHeapUsedBytes", node.javaHeapUsedBytes());

            if (node.byType() != null) {
                var byType = new JsonObject();
                node.byType().forEach(byType::addProperty);
                json.add("byType", byType);
            }
        }

        if (node.children() != null && !node.children().isEmpty()) {
            var children = new JsonArray();
            for (var child : node.children()) {
                children.add(context.serialize(child, DJLMemoryManagement.ManagerNode.class));
            }
            json.add("children", children);
        }

        return json;
    }
}
