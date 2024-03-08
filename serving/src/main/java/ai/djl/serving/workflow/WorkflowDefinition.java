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
import ai.djl.repository.FilenameUtils;
import ai.djl.serving.util.MutableClassLoader;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkerPoolConfig;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * This class is for parsing the JSON or YAML definition for a {@link Workflow}.
 *
 * <p>It can then be converted into a {@link Workflow} using {@link #toWorkflow()}.
 */
public class WorkflowDefinition {

    String name;
    String version;
    String baseUri;

    Map<String, ModelInfo<Input, Output>> models;

    @SerializedName("workflow")
    Map<String, WorkflowExpression> expressions;

    @SerializedName("functions")
    Map<String, String> funcs;

    @SerializedName("configs")
    Map<String, Map<String, Object>> configs;

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
        return parse(null, path.toUri());
    }

    /**
     * Parses a new {@link WorkflowDefinition} from an input stream.
     *
     * @param name the workflow name (null for no name)
     * @param uri the uri of the file
     * @return the parsed {@link WorkflowDefinition}
     * @throws IOException if read from uri failed
     */
    public static WorkflowDefinition parse(String name, URI uri) throws IOException {
        return parse(name, uri, new ConcurrentHashMap<>());
    }

    static WorkflowDefinition parse(String name, URI uri, Map<String, String> templateReplacements)
            throws IOException {
        String type = FilenameUtils.getFileExtension(Objects.requireNonNull(uri.toString()));

        // Default model_dir template replacement
        if (templateReplacements == null) {
            templateReplacements = new ConcurrentHashMap<>();
        }
        templateReplacements.put("model_dir", getWorkflowDir(uri.toString()));

        try (InputStream is = uri.toURL().openStream();
                Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            WorkflowDefinition wd = parse(type, reader, templateReplacements);
            if (name != null) {
                wd.name = name;
            }
            if (wd.baseUri == null) {
                wd.baseUri = uri.toString();
            }
            return wd;
        }
    }

    private static WorkflowDefinition parse(
            String type, Reader input, Map<String, String> templateReplacements) {
        if (templateReplacements != null) {
            String updatedInput =
                    new BufferedReader(input)
                            .lines()
                            .map(
                                    l -> {
                                        for (Entry<String, String> replacement :
                                                templateReplacements.entrySet()) {
                                            l =
                                                    l.replace(
                                                            "{" + replacement.getKey() + "}",
                                                            replacement.getValue());
                                        }
                                        return l;
                                    })
                            .collect(Collectors.joining("\n"));
            input = new StringReader(updatedInput);
        }
        if ("yml".equalsIgnoreCase(type) || "yaml".equalsIgnoreCase(type)) {
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
        } else if ("json".equalsIgnoreCase(type)) {
            return GSON.fromJson(input, WorkflowDefinition.class);
        } else {
            throw new IllegalArgumentException("Unexpected file type: " + type);
        }
    }

    /**
     * Returns the full workflow url if the url points to a workflow definition file.
     *
     * @param link the workflow url
     * @return the workflow URL
     */
    public static URI toWorkflowUri(String link) {
        if (link.startsWith("http") && link.endsWith(".json")
                || link.endsWith(".yml")
                || link.endsWith(".yaml")) {
            return URI.create(link);
        }
        URI uri = URI.create(link);
        String scheme = uri.getScheme();
        if (scheme != null && !"file".equals(scheme)) {
            return null;
        }
        String uriPath = uri.getPath();
        if (uriPath == null) {
            uriPath = uri.getSchemeSpecificPart();
        }
        if (uriPath.startsWith("/") && System.getProperty("os.name").startsWith("Win")) {
            uriPath = uriPath.substring(1);
        }
        Path path = Paths.get(uriPath);
        if (!Files.exists(path)) {
            return null;
        }
        if (uriPath.endsWith(".json") || uriPath.endsWith(".yml") || uriPath.endsWith(".yaml")) {
            return path.toUri();
        }
        if (Files.isDirectory(path)) {
            Path file = path.resolve("workflow.json");
            if (Files.isRegularFile(file)) {
                return file.toUri();
            }
            file = path.resolve("workflow.yml");
            if (Files.isRegularFile(file)) {
                return file.toUri();
            }
            file = path.resolve("workflow.yaml");
            if (Files.isRegularFile(file)) {
                return file.toUri();
            }
        }
        return null;
    }

    /**
     * Converts the {@link WorkflowDefinition} into a workflow.
     *
     * @return a new {@link Workflow} matching this definition
     * @throws BadWorkflowException if the workflow could not be parsed successfully
     */
    public Workflow toWorkflow() throws BadWorkflowException {
        String workflowDir = getWorkflowDir(baseUri);

        if (models != null) {
            for (Entry<String, ModelInfo<Input, Output>> emd : models.entrySet()) {
                ModelInfo<Input, Output> md = emd.getValue();
                md.setId(emd.getKey());
            }
        }

        Map<String, WorkflowFunction> loadedFunctions = new ConcurrentHashMap<>();
        if (funcs != null) {
            String uriPath = URI.create(workflowDir).getPath();
            if (uriPath.startsWith("/") && System.getProperty("os.name").startsWith("Win")) {
                uriPath = uriPath.substring(1);
            }
            Path path = Paths.get(uriPath).resolve("libs");
            Path classDir = path.resolve("classes");
            if (Files.exists(classDir)) {
                ClassLoaderUtils.compileJavaClass(path);
            }
            ClassLoader mcl = MutableClassLoader.getInstance();
            ClassLoader ccl = Thread.currentThread().getContextClassLoader();
            try {
                Thread.currentThread().setContextClassLoader(mcl);
                for (Entry<String, String> f : funcs.entrySet()) {
                    WorkflowFunction func =
                            ClassLoaderUtils.findImplementation(
                                    path, WorkflowFunction.class, f.getValue());
                    if (func == null) {
                        throw new BadWorkflowException("Could not load function " + f.getKey());
                    }
                    loadedFunctions.put(f.getKey(), func);
                }
            } finally {
                Thread.currentThread().setContextClassLoader(ccl);
            }
        }

        Map<String, WorkerPoolConfig<Input, Output>> wpcs = new ConcurrentHashMap<>(models);
        wpcs.putAll(models);
        return new Workflow(name, version, wpcs, expressions, configs, loadedFunctions);
    }

    private static String getWorkflowDir(String uri) {
        int pos = uri.lastIndexOf('/');
        return uri.substring(0, pos);
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
                return new ModelInfo<>(json.getAsString());
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
