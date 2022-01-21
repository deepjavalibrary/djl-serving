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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.serving.workflow.function.WorkflowFunction;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.JsonUtils;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.annotations.SerializedName;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import org.yaml.snakeyaml.Yaml;

/**
 * This class is for parsing the JSON or YAML definition for a {@link Workflow}.
 *
 * <p>It can then be converted into a {@link Workflow} using {@link #toWorkflow()}.
 */
public class WorkflowDefinition {

    String name;
    String version;
    transient String url;

    Map<String, ModelDefinition> models;

    @SerializedName("workflow")
    Map<String, WorkflowExpression> expressions;

    @SerializedName("functions")
    Map<String, String> funcs;

    Integer queueSize;
    Integer maxIdleTime;
    Integer maxBatchDelay;
    Integer batchSize;

    private static final Yaml YAML = new Yaml();
    public static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(ModelDefinition.class, new ModelDefinitionDeserializer())
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
        WorkflowDefinition wd;
        try (Reader reader = Files.newBufferedReader(path)) {
            String fileName = Objects.requireNonNull(path.toString());
            if (fileName.endsWith(".yml") || fileName.endsWith(".yaml")) {
                Object yaml = YAML.load(reader);
                String asJson = GSON.toJson(yaml);
                wd = GSON.fromJson(asJson, WorkflowDefinition.class);
            } else if (fileName.endsWith(".json")) {
                wd = GSON.fromJson(reader, WorkflowDefinition.class);
            } else {
                throw new IllegalArgumentException(
                        "Unexpected file type in workflow file: " + path);
            }
        }
        wd.url = path.toUri().toString();
        return wd;
    }

    /**
     * Converts the {@link WorkflowDefinition} into a workflow.
     *
     * @return a new {@link Workflow} matching this definition
     * @throws ModelNotFoundException if the definition contains an unknown model
     * @throws MalformedModelException if the definition contains a malformed model
     * @throws IOException if it fails to load the definition or resources in it
     * @throws BadWorkflowException if the workflow could not be parsed successfully
     */
    public Workflow toWorkflow()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    BadWorkflowException {
        Map<String, ModelInfo> loadedModels = new ConcurrentHashMap<>();
        if (models != null) {
            for (Entry<String, ModelDefinition> emd : models.entrySet()) {
                ModelDefinition md = emd.getValue();
                ZooModel<Input, Output> model = md.criteria.loadModel();

                ConfigManager configManager = ConfigManager.getInstance();
                int newQueueSize =
                        firstNonNull(md.queueSize, queueSize, configManager.getJobQueueSize());
                int newMaxIdleTime =
                        firstNonNull(md.maxIdleTime, maxIdleTime, configManager.getMaxIdleTime());
                int newMaxBatchDelay =
                        firstNonNull(
                                md.maxBatchDelay, maxBatchDelay, configManager.getMaxBatchDelay());
                int newBatchSize =
                        firstNonNull(md.batchSize, batchSize, configManager.getBatchSize());

                ModelInfo modelInfo =
                        new ModelInfo(
                                model.getName(),
                                md.version,
                                model,
                                newQueueSize,
                                newMaxIdleTime,
                                newMaxBatchDelay,
                                newBatchSize);
                loadedModels.put(emd.getKey(), modelInfo);
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

        return new Workflow(name, version, url, loadedModels, expressions, loadedFunctions);
    }

    private int firstNonNull(Integer... inputs) {
        for (Integer input : inputs) {
            if (input != null) {
                return input;
            }
        }
        return 0;
    }

    private static final class ModelDefinition {

        private Criteria<Input, Output> criteria;
        private String version;

        private Integer queueSize;
        private Integer maxIdleTime;
        private Integer maxBatchDelay;
        private Integer batchSize;

        private ModelDefinition(Criteria<Input, Output> criteria) {
            this.criteria = criteria;
        }
    }

    private static final class ModelDefinitionDeserializer
            implements JsonDeserializer<ModelDefinition> {

        /** {@inheritDoc} */
        @Override
        public ModelDefinition deserialize(
                JsonElement json, Type typeOfT, JsonDeserializationContext context) {
            if (json.isJsonObject()) {
                JsonObject obj = json.getAsJsonObject();
                ModelDefinition md = new ModelDefinition(readCriteria(obj, context));
                md.version = readStringProperty(obj, "version");
                md.queueSize = readIntegerProperty(obj, "queueSize");
                md.maxIdleTime = readIntegerProperty(obj, "maxIdleTime");
                md.maxBatchDelay = readIntegerProperty(obj, "maxBatchDelay");
                md.batchSize = readIntegerProperty(obj, "batchSize");
                return md;
            } else if (json.isJsonPrimitive()) {
                return new ModelDefinition(
                        Criteria.builder()
                                .setTypes(Input.class, Output.class)
                                .optModelUrls(json.getAsString())
                                .build());
            } else {
                throw new JsonParseException(
                        "Unexpected type of model definition: should be Criteria object or URI string");
            }
        }

        private Criteria<Input, Output> readCriteria(
                JsonObject obj, JsonDeserializationContext context) {
            try {
                Criteria.Builder<Input, Output> criteria =
                        Criteria.builder().setTypes(Input.class, Output.class);

                if (obj.has("application")) {
                    criteria.optApplication(Application.of(obj.get("application").getAsString()));
                }
                if (obj.has("engine")) {
                    criteria.optEngine(obj.get("engine").getAsString());
                }
                if (obj.has("groupId")) {
                    criteria.optGroupId(obj.get("groupId").getAsString());
                }
                if (obj.has("artifactId")) {
                    criteria.optArtifactId(obj.get("artifactId").getAsString());
                }
                if (obj.has("modelUrls")) {
                    criteria.optModelUrls(obj.get("modelUrls").getAsString());
                }
                if (obj.has("modelZoo")) {
                    criteria.optModelZoo(ModelZoo.getModelZoo(obj.get("modelZoo").getAsString()));
                }
                if (obj.has("filters")) {
                    Type tp = new TypeToken<Map<String, String>>() {}.getType();
                    criteria.optFilters(context.deserialize(obj.get("filters"), tp));
                }
                if (obj.has("arguments")) {
                    Type tp = new TypeToken<Map<String, Object>>() {}.getType();
                    criteria.optFilters(context.deserialize(obj.get("arguments"), tp));
                }
                if (obj.has("options")) {
                    Type tp = new TypeToken<Map<String, String>>() {}.getType();
                    criteria.optFilters(context.deserialize(obj.get("options"), tp));
                }
                if (obj.has("modelName")) {
                    criteria.optArtifactId(obj.get("modelName").getAsString());
                }
                if (obj.has("translatorFactory")) {
                    Class<? extends TranslatorFactory> clazz =
                            Class.forName(obj.get("translatorFactory").getAsString())
                                    .asSubclass(TranslatorFactory.class);
                    criteria.optTranslatorFactory(clazz.getConstructor().newInstance());
                }
                if (obj.has("translator")) {
                    Class<? extends ServingTranslator> clazz =
                            Class.forName(obj.get("translator").getAsString())
                                    .asSubclass(ServingTranslator.class);
                    criteria.optTranslator(clazz.getConstructor().newInstance());
                }

                return criteria.build();
            } catch (ClassNotFoundException
                    | InvocationTargetException
                    | InstantiationException
                    | IllegalAccessException
                    | NoSuchMethodException e) {
                throw new JsonParseException("Failed to parse model definition", e);
            }
        }

        private String readStringProperty(JsonObject obj, String name) {
            if (obj.has(name)) {
                return obj.get(name).getAsString();
            }
            return null;
        }

        private Integer readIntegerProperty(JsonObject obj, String name) {
            if (obj.has(name)) {
                return obj.get(name).getAsInt();
            }
            return null;
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
