/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.workflow;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.serving.workflow.function.WorkflowFunction;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.JsonUtils;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.annotations.SerializedName;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class is for parsing the JSON or YAML definition for a {@link Workflow}.
 *
 * <p>It can then be converted into a {@link Workflow} using {@link #toWorkflow()}.
 */
public class WorkflowDefinition {

    String name;
    String version;

    Map<String, ModelInfo<Input, Output>> models;

    @SerializedName("workflow")
    Map<String, WorkflowExpression> expressions;

    @SerializedName("functions")
    Map<String, String> funcs;

    int queueSize;
    int maxIdleTime;
    int maxBatchDelay;
    int batchSize;

    public static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(ModelInfo.class, new ModelDefinitionDeserializer())
                    .registerTypeAdapter(WorkflowExpression.class, new ExpressionDeserializer())
                    .registerTypeAdapter(Item.class, new ExpressionItemDeserializer())
                    .create();

    /**
     * Parses a new {@link WorkflowDefinition} from a file path.
     *
     * @param path the path to parse the file from
     * @return the parsed {@link WorkflowDefinition}
     * @throws IOException if it fails to load the file for parsing
     */
    public static WorkflowDefinition parse(Path path) throws IOException {
        return parse(path.toUri(), Files.newBufferedReader(path));
    }

    /**
     * Parses a new {@link WorkflowDefinition} from an input stream.
     *
     * @param uri the uri of the file
     * @param input the input
     * @return the parsed {@link WorkflowDefinition}
     */
    public static WorkflowDefinition parse(URI uri, InputStream input) {
        return parse(uri, new InputStreamReader(input, StandardCharsets.UTF_8));
    }

    /**
     * Parses a new {@link WorkflowDefinition} from a reader.
     *
     * @param uri the uri of the file
     * @param input the input
     * @return the parsed {@link WorkflowDefinition}
     */
    public static WorkflowDefinition parse(URI uri, Reader input) {
        String fileName = Objects.requireNonNull(uri.toString());
        if (fileName.endsWith(".yml") || fileName.endsWith(".yaml")) {
            try {
                ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
                Class<?> clazz = Class.forName("org.yaml.snakeyaml.Yaml", true, cl);
                Constructor<?> constructor = clazz.getConstructor();
                Method method = clazz.getMethod("load", Reader.class);
                Object obj = constructor.newInstance();
                Object yaml = method.invoke(obj, input);
                String asJson = GSON.toJson(yaml);
                return GSON.fromJson(asJson, WorkflowDefinition.class);
            } catch (ReflectiveOperationException e) {
                throw new IllegalArgumentException(
                        "Yaml parsing is not supported. In order to support parsing Yaml files, the"
                                + " dependency snakeyaml is required. Please add"
                                + " 'org.yaml.snakeyaml.Yaml' to your classpath, pom.xml, or"
                                + " build.gradle.",
                        e);
            }
        } else if (fileName.endsWith(".json")) {
            return GSON.fromJson(input, WorkflowDefinition.class);
        } else {
            throw new IllegalArgumentException("Unexpected file type in workflow file: " + uri);
        }
    }

    /**
     * Converts the {@link WorkflowDefinition} into a workflow.
     *
     * @return a new {@link Workflow} matching this definition
     * @throws BadWorkflowException if the workflow could not be parsed successfully
     */
    public Workflow toWorkflow() throws BadWorkflowException {
        if (models != null) {
            WlmConfigManager wlmc = WlmConfigManager.getInstance();
            for (Entry<String, ModelInfo<Input, Output>> emd : models.entrySet()) {
                ModelInfo<Input, Output> md = emd.getValue();
                md.setModelId(emd.getKey());
                md.setQueueSize(firstValid(md.getQueueSize(), queueSize, wlmc.getJobQueueSize()));
                md.setMaxIdleTime(
                        firstValid(md.getMaxIdleTime(), maxIdleTime, wlmc.getMaxIdleTime()));
                md.setMaxBatchDelay(
                        firstValid(md.getMaxBatchDelay(), maxBatchDelay, wlmc.getMaxBatchDelay()));
                md.setBatchSize(firstValid(md.getBatchSize(), batchSize, wlmc.getBatchSize()));
                if (name == null) {
                    name = emd.getKey();
                }
            }
        }

        Map<String, WorkflowFunction> loadedFunctions = new ConcurrentHashMap<>();
        if (funcs != null) {
            for (Entry<String, String> f : funcs.entrySet()) {
                try {
                    Class<? extends WorkflowFunction> clazz =
                            Class.forName(f.getValue()).asSubclass(WorkflowFunction.class);
                    loadedFunctions.put(f.getKey(), clazz.getConstructor().newInstance());
                } catch (Exception e) {
                    throw new BadWorkflowException("Could not load function " + f.getKey(), e);
                }
            }
        }

        return new Workflow(name, version, models, expressions, loadedFunctions);
    }

    private int firstValid(int... inputs) {
        for (int input : inputs) {
            if (input > 0) {
                return input;
            }
        }
        return 0;
    }

    private static final class ModelDefinitionDeserializer
            implements JsonDeserializer<ModelInfo<Input, Output>> {

        /** {@inheritDoc} */
        @SuppressWarnings("unchecked")
        @Override
        public ModelInfo<Input, Output> deserialize(
                JsonElement json, Type typeOfT, JsonDeserializationContext context) {
            if (json.isJsonObject()) {
                ModelInfo<Input, Output> model = JsonUtils.GSON.fromJson(json, ModelInfo.class);
                model.hasInputOutputClass(Input.class, Output.class);
                return model;
            } else if (json.isJsonPrimitive()) {
                return new ModelInfo<>(json.getAsString(), Input.class, Output.class);
            }
            throw new JsonParseException(
                    "Unexpected type of model definition: should be Criteria object or URI string");
        }
    }

    private static final class ExpressionDeserializer
            implements JsonDeserializer<WorkflowExpression> {

        /** {@inheritDoc} */
        @Override
        public WorkflowExpression deserialize(
                JsonElement json, Type typeOfT, JsonDeserializationContext context) {
            JsonArray array = json.getAsJsonArray();
            List<Item> args = new ArrayList<>(array.size());
            for (JsonElement el : array) {
                args.add(context.deserialize(el, Item.class));
            }
            return new WorkflowExpression(args);
        }
    }

    private static final class ExpressionItemDeserializer implements JsonDeserializer<Item> {

        /** {@inheritDoc} */
        @Override
        public Item deserialize(
                JsonElement json, Type typeOfT, JsonDeserializationContext context) {
            if (json.isJsonArray()) {
                return new Item(
                        (WorkflowExpression) context.deserialize(json, WorkflowExpression.class));
            } else if (json.isJsonPrimitive()) {
                return new Item(json.getAsString());
            } else {
                throw new JsonParseException("Unexpected JSON element in expression item");
            }
        }
    }
}
