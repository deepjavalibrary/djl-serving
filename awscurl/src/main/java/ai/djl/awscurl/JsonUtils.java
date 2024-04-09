/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.awscurl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

final class JsonUtils {

    static final Gson GSON = new Gson();
    static final Gson GSON_PRETTY = new GsonBuilder().setPrettyPrinting().create();
    private static AtomicBoolean printException = new AtomicBoolean(true);

    private JsonUtils() {}

    static void getJsonList(JsonElement element, List<String> list, String[] names) {
        if (element.isJsonArray()) {
            JsonArray array = element.getAsJsonArray();
            for (int i = 0; i < array.size(); ++i) {
                getJsonList(array.get(i), list, names);
            }
        } else if (element.isJsonObject()) {
            JsonObject obj = element.getAsJsonObject();
            JsonElement e;
            if (names == null) {
                e = find(obj, new String[] {"token", "text"}, 0);
                if (e == null) {
                    e = obj.get("generated_text");
                }
            } else {
                e = find(obj, names, 0);
            }
            if (e != null) {
                if (e.isJsonPrimitive()) {
                    list.add(e.getAsString());
                } else if (e.isJsonArray()) {
                    JsonArray array = e.getAsJsonArray();
                    for (int i = 0; i < array.size(); ++i) {
                        JsonElement text = array.get(i);
                        if (text.isJsonPrimitive()) {
                            list.add(text.getAsString());
                        } else {
                            AwsCurl.logger.debug("Ignore element: {}", text);
                        }
                    }
                } else {
                    AwsCurl.logger.debug("Ignore element: {}", e);
                }
            }
        } else {
            AwsCurl.logger.debug("Ignore element: {}", element);
        }
    }

    static void resetException() {
        printException.set(true);
    }

    @SuppressWarnings("PMD.SystemPrintln")
    static boolean processJsonLine(
            List<StringBuilder> list, OutputStream ps, String line, String[] name)
            throws IOException {
        boolean hasError = false;
        try {
            JsonObject map = GSON.fromJson(line, JsonObject.class);
            JsonElement outputs;
            if (name == null) {
                outputs = map.get("outputs");
                if (outputs == null) {
                    outputs = find(map, new String[] {"token", "text"}, 0);
                    if (outputs == null) {
                        outputs = map.get("generated_text");
                    }
                }
            } else {
                outputs = find(map, name, 0);
            }
            if (outputs != null) {
                if (outputs.isJsonArray()) {
                    JsonArray arr = outputs.getAsJsonArray();
                    List<JsonElement> items = arr.asList();
                    if (list.isEmpty()) {
                        for (JsonElement s : items) {
                            list.add(new StringBuilder(s.getAsString()));
                        }
                    } else {
                        for (int i = 0; i < items.size(); ++i) {
                            list.get(i).append(items.get(i).getAsString());
                        }
                    }
                } else if (outputs.isJsonPrimitive()) {
                    if (list.isEmpty()) {
                        list.add(new StringBuilder(outputs.getAsString()));
                    } else {
                        list.get(0).append(outputs.getAsString());
                    }
                }
            }
        } catch (JsonParseException e) {
            if (printException.getAndSet(false)) {
                System.out.println("Invalid json line: " + line);
                AwsCurl.logger.debug("Invalid json line", e);
            }
            hasError = true;
        }

        ps.write(line.getBytes(StandardCharsets.UTF_8));
        ps.write(new byte[] {'\n'});
        return hasError;
    }

    static JsonElement find(JsonElement root, String[] names, int level) {
        if (root != null && level <= names.length) {
            if (root.isJsonObject()) {
                JsonObject obj = root.getAsJsonObject();
                JsonElement ret = obj.get(names[level]);
                return find(ret, names, ++level);
            } else if (root.isJsonArray()) {
                JsonArray array = root.getAsJsonArray();
                for (int i = 0; i < array.size(); ++i) {
                    JsonElement element = array.get(i);
                    JsonElement ret = find(element, names, level);
                    if (ret != null) {
                        return ret;
                    }
                }
            } else if (level == names.length) {
                return root;
            }
        }
        return null;
    }
}
