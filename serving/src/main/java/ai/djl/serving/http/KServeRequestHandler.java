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
package ai.djl.serving.http;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(KServeRequestHandler.class);

    private static final Pattern PATTERN = Pattern.compile("^/v2/.+");
    private static final String EMPTY_BODY = "";

    private RequestParser requestParser;

    public KServeRequestHandler() {
        this.requestParser = new RequestParser();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        HttpMethod method = req.method();
        try {
            if (isKServeDescribeModelReq(segments)) {
                isHttpGetRequestOrThrowException(method);
                handleKServeDescribeModel(ctx, segments);
            } else if (isKServeDescribeHealthReadyReq(segments)
                    || isKServeDescribeHealthLiveReq(segments)) {
                isHttpGetRequestOrThrowException(method);
                handleKServeDescribeHealth(ctx);
            } else if (isKServeDescribeModelReadyReq(segments)) {
                isHttpGetRequestOrThrowException(method);
                handleKServeDescribeModelReady(ctx, segments);
            } else if (isKserveDescribeInferenceReq(segments,method)) {
                handleKServeDescribeInfer(ctx, req, decoder, segments);
            } else {
                throw new ResourceNotFoundException();
            }
        } catch (Exception exception) {
            onException(exception, ctx);
        }
    }

    private void isHttpGetRequestOrThrowException(HttpMethod method) {
        if (!HttpMethod.GET.equals(method)) {
            throw new MethodNotAllowedException();
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments) {
        return (segments.length == 4 || (segments.length == 6 && "version".equals(segments[4])))
                && "models".equals(segments[2]);
    }

    private boolean isKServeDescribeHealthLiveReq(String[] segments) {
        return segments.length == 4 && "health".equals(segments[2]) && "live".equals((segments[3]));
    }

    private boolean isKServeDescribeHealthReadyReq(String[] segments) {
        return segments.length == 4
                && "health".equals(segments[2])
                && "ready".equals((segments[3]));
    }

    private boolean isKServeDescribeModelReadyReq(String[] segments) {
        return (segments.length == 5 || (segments.length == 7 && "version".equals(segments[4])))
                && "models".equals(segments[2])
                && "ready".equals(segments[segments.length - 1]);
    }

    private boolean isKserveDescribeInferenceReq(String[] segments, HttpMethod method) {
        return "v2".equals(segments[1])
                && "models".equals(segments[2])
                && HttpMethod.POST.equals(method)
                && "infer".equals(segments[segments.length - 1]);
    }

    private void handleKServeDescribeModel(ChannelHandlerContext ctx, String[] segments)
            throws ModelNotFoundException {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 4) {
            modelVersion = segments[5];
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        Endpoint endpoint = endpoints.get(modelName);
        List<Workflow> workflows;
        if (endpoint != null) {
            workflows = endpoint.getWorkflows();
        } else {
            workflows = Collections.emptyList();
        }

        // TODO: How to handle multiple models and version?
        KServeDescribeModelResponse response = new KServeDescribeModelResponse();
        List<String> versions = new ArrayList<>();
        response.setVersions(versions);
        Model model = null;
        for (Workflow wf : workflows) {
            String version = wf.getVersion();
            if (modelVersion != null && !modelVersion.equals(version)) {
                continue;
            }
            if (version != null) {
                // TODO: null is a valid version in DJL
                versions.add(version);
            }
            if (model != null) {
                // only add one model
                continue;
            }

            for (ModelInfo<Input, Output> modelInfo : wf.getModels()) {
                if (modelInfo.getStatus() == ModelInfo.Status.READY) {
                    response.setName(wf.getName());
                    response.setPlatformForEngineName(modelInfo.getEngineName());
                    Device device = modelInfo.getModels().keySet().iterator().next();
                    model = modelInfo.getModel(device);

                    DataType dataType = model.getDataType();
                    if (model.describeInput() != null) {
                        for (Pair<String, Shape> input : model.describeInput()) {
                            response.addInput(input.getKey(), dataType, input.getValue());
                        }
                    }

                    if (model.describeOutput() != null) {
                        for (Pair<String, Shape> output : model.describeOutput()) {
                            response.addOutput(output.getKey(), dataType, output.getValue());
                        }
                    }
                    break;
                }
            }
        }
        if (model == null) {
            throw new ModelNotFoundException(
                    "Model not found: "
                            + modelName
                            + (modelVersion == null ? "" : '/' + modelVersion));
        }

        NettyUtils.sendJsonResponse(ctx, response);
    }

    private void handleKServeDescribeHealth(ChannelHandlerContext ctx) {
        ModelManager.getInstance()
                .workerStatus()
                .thenAccept(
                        w -> {
                            boolean hasFailure = (boolean) w.get("hasFailure");
                            boolean hasPending = (boolean) w.get("hasPending");

                            HttpResponseStatus httpResponseStatus;
                            if (hasFailure || hasPending) {
                                httpResponseStatus = HttpResponseStatus.FAILED_DEPENDENCY;
                            } else {
                                httpResponseStatus = HttpResponseStatus.OK;
                            }
                            NettyUtils.sendJsonResponse(ctx, EMPTY_BODY, httpResponseStatus);
                        });
    }

    private void handleKServeDescribeModelReady(ChannelHandlerContext ctx, String[] segments)
            throws ModelNotFoundException {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 5) {
            modelVersion = segments[5];
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(modelName, modelVersion);

        ModelInfo.Status status = modelInfo.getStatus();
        HttpResponseStatus httpResponseStatus;
        if (status == ModelInfo.Status.READY) {
            httpResponseStatus = HttpResponseStatus.OK;
        } else {
            httpResponseStatus = HttpResponseStatus.FAILED_DEPENDENCY;
        }
        NettyUtils.sendJsonResponse(ctx, EMPTY_BODY, httpResponseStatus);
    }

    private ModelInfo<Input, Output> getModelInfo(String modelName, String modelVersion)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);
        Collection<ModelInfo<Input, Output>> models;
        if (workflow != null) {
            models = workflow.getModels();
        } else {
            models = Collections.emptyList();
        }
        if (models.isEmpty()) {
            throw new ModelNotFoundException(
                    "Model not found: "
                            + modelName
                            + (modelVersion == null ? "" : '/' + modelVersion));
        }
        return models.iterator().next();
    }

    private void handleKServeDescribeInfer(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 5) {
            modelVersion = segments[5];
        }
        Input inferenceRequest = requestParser.parseRequest(req, decoder);
        infer(ctx, inferenceRequest, modelName, modelVersion);
    }

    private ArrayList<KserveIO> getInputListFromString(String requestString) {
        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        Gson gson = builder.create();
        return gson.fromJson(requestString, new TypeToken<ArrayList<KserveIO>>() {}.getType());
    }

    private ArrayList<RequestOutput> getOutputListFromString(String requestString) {
        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        Gson gson = builder.create();
        ArrayList<RequestOutput> arrayList =
                gson.fromJson(
                        requestString, new TypeToken<ArrayList<RequestOutput>>() {}.getType());
        return arrayList;
    }

    private void infer(
            ChannelHandlerContext ctx,
            Input inferenceRequest,
            String workflowName,
            String version) {
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(workflowName, version, true);
        if (workflow == null) {
            sendOutput(new Output(HttpResponseStatus.BAD_REQUEST.code(), ""), ctx);
        }

        String requestID = inferenceRequest.get("id").getAsString();
        String requestInputsString = inferenceRequest.get("inputs").getAsString();
        String requestOutputsString = inferenceRequest.get("outputs").getAsString();

        // transform string to json object
        ArrayList<KserveIO> inputsArrayList = getInputListFromString(requestInputsString);
        ArrayList<RequestOutput> outputsArrayList = getOutputListFromString(requestOutputsString);

        Input input = new Input();
        // construct the  input
        try (NDManager manager = NDManager.newBaseManager()) {
            NDList list = new NDList();
            for (KserveIO requestInput : inputsArrayList) {
                List<Double> dataList = requestInput.getData();
                double[] dataArray = dataList.stream().mapToDouble(j -> j).toArray();

                Shape shape = new Shape(requestInput.getShape());
                NDArray shapeNDArray = manager.create(shape);
                shapeNDArray.setName("shape");
                NDArray dataNDArray = manager.create(dataArray);
                dataNDArray.setName("data");

                list.add(shapeNDArray);
                list.add(dataNDArray);
            }
            // here must be getAsBytes because list will die after the lifecycle
            input.add("data", list.getAsBytes());
        }
        runJob(modelManager, ctx, workflow, input, outputsArrayList, requestID);
    }

    void runJob(
            ModelManager modelManager,
            ChannelHandlerContext ctx,
            Workflow workflow,
            Input input,
            ArrayList<RequestOutput> outputs,
            String requestID) {
        modelManager
                .runJob(workflow, input)
                .whenComplete(
                        (o, t) -> {
                            if (o != null) {
                                responseOutput(outputs, requestID, o, ctx);
                            }
                        })
                .exceptionally(
                        t -> {
                            onModelException(t.getCause(), ctx);
                            return null;
                        });
    }

    void responseOutput(
            ArrayList<RequestOutput> requestOutputs,
            String requestID,
            Output output,
            ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        int code = output.getCode();
        if (code == 200) {
            status = HttpResponseStatus.OK;
        } else {
            if (code >= 500) {
                status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
            } else if (code >= 400) {
                status = HttpResponseStatus.BAD_REQUEST;
            } else {
                status = new HttpResponseStatus(code, output.getMessage());
            }
        }

        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            resp.headers().set(entry.getKey(), entry.getValue());
        }
        // construct the output
        BytesSupplier data = output.getData();
        String datatype =
                output.get("datatype") != null ? output.get("datatype").getAsString() : "null";
//        String shape = output.get("shape") != null ? output.get("shape").getAsString() : "null";
//        String responseData =
//                output.get("data") != null ? output.get("data").getAsString() : "null";

        ArrayList<KserveIO> responseOutputList = new ArrayList<>();
        for (RequestOutput requestOutput : requestOutputs) {
            KserveIO responseOutput = new KserveIO();
            responseOutput.setName(requestOutput.getName());
            responseOutput.setDatatype(datatype);
//            responseOutput.setShape(shape);
//            responseOutput.setData(responseData);

            responseOutputList.add(responseOutput);
        }
        InferenceResponse inferenceResponse = new InferenceResponse();
        inferenceResponse.setId(requestID);
        inferenceResponse.setList(responseOutputList);

        String json = new Gson().toJson(inferenceResponse);

        if (data != null) {
            resp.content().writeBytes(json.getBytes());
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
    }

    void sendOutput(Output output, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        int code = output.getCode();
        if (code == 200) {
            status = HttpResponseStatus.OK;
        } else {
            if (code >= 500) {
                status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
            } else if (code >= 400) {
                status = HttpResponseStatus.BAD_REQUEST;
            } else {
                status = new HttpResponseStatus(code, output.getMessage());
            }
        }

        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            resp.headers().set(entry.getKey(), entry.getValue());
        }
        BytesSupplier data = output.getData();

        if (data != null) {
            resp.content().writeBytes(data.getAsBytes());
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
    }

    private void onException(Exception ex, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (ex instanceof ModelNotFoundException) {
            status = HttpResponseStatus.NOT_FOUND;
        } else if (ex instanceof MethodNotAllowedException) {
            status = HttpResponseStatus.METHOD_NOT_ALLOWED;
        } else {
            logger.warn("Unexpected error", ex);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        Map<String, String> content = new ConcurrentHashMap<>();
        content.put("error", ex.getMessage());

        NettyUtils.sendJsonResponse(ctx, content, status);
    }

    void onModelException(Throwable t, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (t instanceof TranslateException) {
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmException) {
            logger.warn(t.getMessage(), t);
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", t);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }
        if (ctx != null) {
            NettyUtils.sendError(ctx, status, t);
        }
    }


    public static final class KserveIO {
        private String name;
        private List<Long> shape;
        private String datatype;
        // TODO: accept all types of data
        private List<Double> data;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public List<Long> getShape() {
            return shape;
        }

        public void setShape(List<Long> shape) {
            this.shape = shape;
        }

        public String getDatatype() {
            return datatype;
        }

        public void setDatatype(String datatype) {
            this.datatype = datatype;
        }

        public List<Double> getData() {
            return data;
        }

        public void setData(List<Double> data) {
            this.data = data;
        }

        public String toString() {
            return "Input [ name: "
                    + name
                    + ", datatype: "
                    + datatype
                    + ", shape: "
                    + shape
                    + ", data "
                    + data
                    + " ]";
        }
    }

    public static final class RequestOutput {
        private String name;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String toString() {
            return "Output [ name: " + name + " ]";
        }
    }

    public static final class InferenceResponse {
        private ArrayList<KserveIO> outputs;
        private String id;

        public void setId(String id) {
            this.id = id;
        }

        public void setList(ArrayList<KserveIO> responseOutput) {
            this.outputs = responseOutput;
        }
    }

    public static final class RequestInput {
        private String name;
        private String shape;
        private String datatype;
        private String data;

        public void setName(String name) {
            this.name = name;
        }

        public void setShape(String shape) {
            this.shape = shape;
        }

        public void setDatatype(String datatype) {
            this.datatype = datatype;
        }

        public void setData(String data) {
            this.data = data;
        }
    }
}
