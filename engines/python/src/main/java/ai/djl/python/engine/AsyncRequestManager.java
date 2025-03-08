/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.python.engine;

import ai.djl.Model;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metrics;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.RandomUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

class AsyncRequestManager {

    private static final Logger logger = LoggerFactory.getLogger(AsyncRequestManager.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");
    private static final String REQUEST_TRACKING_ID = "request_tracking_id";

    private Dimension dimension;
    private Metrics metrics;
    private ConcurrentHashMap<String, Request> activeRequests;
    private PyProcess process;

    AsyncRequestManager(PyProcess process, Model model) {
        this.dimension = new Dimension("Model", model.getProperty("metric_dimension", "model"));
        this.activeRequests = new ConcurrentHashMap<>();
        this.process = process;
        if (Boolean.parseBoolean(model.getProperty("log_request_metric"))) {
            int metricsAggregation = model.intProperty("metrics_aggregation", 1000);
            metrics = new Metrics();
            metrics.setLimit(metricsAggregation);
            metrics.setOnLimit(
                    (m, s) -> {
                        MODEL_METRIC.info("{}", m.percentile(s, 50));
                        MODEL_METRIC.info("{}", m.percentile(s, 90));
                    });
        }
    }

    Output addInput(Input input) {
        String requestTrackingId = UUID.randomUUID().toString();
        input.addProperty(REQUEST_TRACKING_ID, requestTrackingId);
        String seed = String.valueOf(RandomUtils.nextInt());
        Request request = new Request(input, seed, metrics, dimension);
        activeRequests.put(requestTrackingId, request);
        logger.debug(
                "Adding continuous batch request with external requestId {}, internal trackingId"
                        + " {}",
                request.getRequestId(),
                requestTrackingId);
        process.sendRequest(input);
        return request.output;
    }

    void addOutput(Output output) {
        // This check is here to handle the empty inference request used to load the model.
        // Refactor to make this check not needed, or add validation to ensure it only occurs on
        // the initial model load.
        if (!this.process.isReady()) {
            logger.info("process is not ready");
            return;
        }
        PairList<String, BytesSupplier> content = output.getContent();
        assert content.size() == 1;
        Map<String, String> prop = output.getProperties();
        byte[] responseContent = content.get(0).getValue().getAsBytes();
        String requestTrackingId = prop.get(REQUEST_TRACKING_ID);
        Request request = activeRequests.get(requestTrackingId);
        request.addResponse(responseContent, prop);
        if (request.last) {
            logger.info("Request [{}] completed", request.getRequestId());
            logger.debug("Removing request with trackingId {}", requestTrackingId);
            activeRequests.remove(requestTrackingId);
        }
    }

    void terminateInFlightRequests() {
        Output output = new Output(500, "Inference Engine Failure");
        BytesSupplier error = BytesSupplier.wrap(JsonUtils.GSON.toJson(output));
        logger.info(
                "Python Engine failed while {} requests are still active", activeRequests.size());
        for (Map.Entry<String, Request> entry : activeRequests.entrySet()) {
            String internalId = entry.getKey();
            Request request = entry.getValue();
            request.last = true;
            request.output.setCode(500);
            request.data.appendContent(error, true);
            logger.debug(
                    "Terminating request with internal Id {}, external Id {}",
                    internalId,
                    request.getRequest());
        }
        activeRequests.clear();
        logger.info("In-flight requests terminated");
    }
}
