/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ModelException;
import ai.djl.metric.Metric;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);
    private static final Pattern PATTERN = Pattern.compile("^/v2([/?].*)?");
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");
    private static final Metric RESPONSE_2_XX = new Metric("2XX", 1);
    private static final Metric RESPONSE_4_XX = new Metric("4XX", 1);
    private static final Metric RESPONSE_5_XX = new Metric("5XX", 1);
    private static final Metric WLM_ERROR = new Metric("WlmError", 1);
    private static final Metric SERVER_ERROR = new Metric("ServerError", 1);
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

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        HttpMethod method = req.method();
        if (!(HttpMethod.GET.equals(method) || HttpMethod.POST.equals(method))) {
            sendOutput(new Output(HttpResponseStatus.METHOD_NOT_ALLOWED.code(), ""), ctx);
            return;
        }
        if (HttpMethod.GET.equals(method) && isKServeDescribeModelReq(segments, method)) {
            handleKServeDescribeModel(ctx, segments);
        } else if (isKServeDescribeHealthReq(segments)) {
            handleKServeDescribeHealth(ctx, segments, method);
        } else if (isKserveDescribeInferenceReq(segments, method)) {
            handleKServeDescribeInfer(ctx, req, decoder, segments);
        } else {
            throw new AssertionError("Invalid request uri: " + req.uri());
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments, HttpMethod method) {
        return HttpMethod.GET.equals(method)
                && (segments.length == 4 || (segments.length == 6 && "version".equals(segments[4])))
                && "v2".equals(segments[1])
                && "models".equals(segments[2]);
    }

    private boolean isKServeDescribeHealthReq(String[] segments) {
        return "v2".equals(segments[1])
                && ((segments.length == 4 && "health".equals(segments[2]))
                        || ("models".equals(segments[2])
                                && "ready".equals(segments[segments.length - 1])));
    }

    private boolean isKserveDescribeInferenceReq(String[] segments, HttpMethod method) {
        return "v2".equals(segments[1])
                && "models".equals(segments[2])
                && HttpMethod.POST.equals(method)
                && "infer".equals(segments[segments.length - 1]);
    }

    private void handleKServeDescribeModel(ChannelHandlerContext ctx, String[] segments) {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 4) {
            modelVersion = segments[5];
        }
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);
        // TODO: Search all workflows if model is not found here.
        ModelInfo<Input, Output> modelInfo =
                workflow.getModels().stream()
                        .filter(
                                model ->
                                        modelName.equals(
                                                model.getModel(model.withDefaultDevice(null))
                                                        .getName()))
                        .findAny()
                        .get();

        KServeDescribeModelResponse response = new KServeDescribeModelResponse();
        // TODO: Include all versions
        response.addModelVersion(modelInfo.getVersion());

        response.setPlatformForEngineName(modelInfo.getEngineName());

        ZooModel<Input, Output> model = modelInfo.getModel(modelInfo.withDefaultDevice(null));
        response.setName(model.getName());
        DataType dataType = model.getDataType();

        for (Pair<String, Shape> input : model.describeInput()) {
            response.addInput(input.getKey(), dataType, input.getValue());
        }

        if (null != model.describeOutput()) {
            for (Pair<String, Shape> output : model.describeOutput()) {
                response.addOutput(output.getKey(), dataType, output.getValue());
            }
        }

        NettyUtils.sendJsonResponse(ctx, response);
    }

    private void handleKServeDescribeHealth(
            ChannelHandlerContext ctx, String[] segments, HttpMethod method) {
        if (!HttpMethod.GET.equals(method)) {
            sendOutput(new Output(HttpResponseStatus.METHOD_NOT_ALLOWED.code(), ""), ctx);
            return;
        }
        if ("models".equals(segments[2])) {
            String modelName = segments[3];
            String modelVersion = null;
            if (segments.length > 5) {
                modelVersion = segments[5];
            }
            ModelManager modelManager = ModelManager.getInstance();
            Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);
            if (workflow == null) {
                sendOutput(new Output(HttpResponseStatus.BAD_REQUEST.code(), ""), ctx);
                return;
            }
            modelManager
                    .modelStatus(modelName, modelVersion)
                    .thenAccept(r -> NettyUtils.sendHttpResponse(ctx, r, true));
        } else {
            switch (segments[3]) {
                case "ready":
                case "live":
                    ModelManager.getInstance()
                            .healthStatus()
                            .thenAccept(r -> NettyUtils.sendHttpResponse(ctx, r, true));
                    break;
                    // if the string is not ready or live, send badrequest code
                default:
                    sendOutput(new Output(HttpResponseStatus.BAD_REQUEST.code(), ""), ctx);
            }
        }
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

    private void infer(
            ChannelHandlerContext ctx,
            Input inferenceRequest,
            String workflowName,
            String version) {
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(workflowName, version, true);

        PairList<String, BytesSupplier> Body = inferenceRequest.getContent();

        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        Gson gson = builder.create();
        ArrayList<RequestInput> inputArrayList =
                gson.fromJson(
                        Body.get("inputs").getAsString(),
                        new TypeToken<ArrayList<RequestInput>>() {}.getType());
        ArrayList<RequestOutput> outputArrayList =
                gson.fromJson(
                        Body.get("outputs").getAsString(),
                        new TypeToken<ArrayList<RequestOutput>>() {}.getType());

        //        byte[] buf = Body.get("inputs").getAsBytes();
        //        System.out.println("buf: " + buf);
        //        try (NDManager manger = NDManager.newBaseManager()) {
        //            NDList ndList = NDList.decode(manger, buf);
        //            for (NDArray array : ndList) {
        //                String name = array.getName();
        //                if (name == null) {
        //                    name = "output__" + ndList.iterator();
        //                }
        //                Shape shape = array.getShape();
        //                DataType type = array.getDataType();
        //                ByteBuffer bb = array.toByteBuffer();
        //
        //            }
        //        }

        if (workflow == null) {
            // TODO: send error msg
        }
        for (int i = 0; i < inputArrayList.size(); i++) {
            // remove the data in request_input which is not the Kserve-needed data
            // then put the real data in that.

            //            Body.remove("data");
            // TODO: the value of data should be NDlist so that it will be accepted by the
            // no-translator model
            //            Body.add("data", BytesSupplier.wrapAsJson(inputArrayList.get(i)));

            //            inferenceRequest.setContent(Body);

            runJob(
                    modelManager,
                    ctx,
                    workflow,
                    inferenceRequest,
                    inputArrayList.get(i),
                    outputArrayList.get(i));

            //            System.out.println("inferenceRequest.get(\"data\")" +
            // inferenceRequest.get("data").getAsString());
        }
    }

    void runJob(
            ModelManager modelManager,
            ChannelHandlerContext ctx,
            Workflow workflow,
            Input inferenceRequest,
            RequestInput input,
            RequestOutput output) {
        modelManager
                .runJob(workflow, inferenceRequest)
                .whenComplete(
                        (o, t) -> {
                            if (o != null) {
                                sendOutput(o, ctx);
                            }
                        })
                .exceptionally(
                        t -> {
                            onException(t.getCause(), ctx);
                            return null;
                        });
    }

    void sendOutput(Output output, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        int code = output.getCode();
        if (code == 200) {
            status = HttpResponseStatus.OK;
            SERVER_METRIC.info("{}", RESPONSE_2_XX);
        } else {
            if (code >= 500) {
                SERVER_METRIC.info("{}", RESPONSE_5_XX);
            } else if (code >= 400) {
                SERVER_METRIC.info("{}", RESPONSE_4_XX);
            } else {
                SERVER_METRIC.info("{}", RESPONSE_2_XX);
            }
            status = new HttpResponseStatus(code, output.getMessage());
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

    void onException(Throwable t, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (t instanceof TranslateException) {
            SERVER_METRIC.info("{}", RESPONSE_4_XX);
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmException) {
            logger.warn(t.getMessage(), t);
            SERVER_METRIC.info("{}", RESPONSE_5_XX);
            SERVER_METRIC.info("{}", WLM_ERROR);
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", t);
            SERVER_METRIC.info("{}", RESPONSE_5_XX);
            SERVER_METRIC.info("{}", SERVER_ERROR);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendError(ctx, status, t);
        }
    }
}

class RequestInput {
    private String name;
    private List<Long> shape;
    private String datatype;
    private List<?> data;

    public RequestInput() {}

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<Long> getShape() {
        return shape;
    }

    public void setShape(List<Long> data) {
        this.shape = shape;
    }

    public String getDatatype() {
        return datatype;
    }

    public void setDatatype(String datatype) {
        this.datatype = datatype;
    }

    public List<?> getData() {
        return data;
    }

    public void setData(List<?> data) {
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

class RequestOutput {
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
