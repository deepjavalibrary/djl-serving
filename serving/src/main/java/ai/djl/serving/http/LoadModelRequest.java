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
package ai.djl.serving.http;

import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;

import com.google.gson.annotations.SerializedName;

import io.netty.handler.codec.http.QueryStringDecoder;

class LoadModelRequest {

    static final String URL = "url";
    static final String DEVICE = "device";
    static final String MAX_WORKER = "max_worker";
    static final String MIN_WORKER = "min_worker";
    static final String SYNCHRONOUS = "synchronous";

    private static final String JOB_QUEUE_SIZE = "job_queue_size";
    private static final String BATCH_SIZE = "batch_size";
    private static final String MODEL_NAME = "model_name";
    private static final String MODEL_VERSION = "model_version";
    private static final String ENGINE_NAME = "engine";
    private static final String MAX_BATCH_DELAY = "max_batch_delay";
    private static final String MAX_IDLE_TIME = "max_idle_time";

    @SerializedName(URL)
    private String modelUrl;

    @SerializedName(MODEL_NAME)
    private String modelName;

    @SerializedName(MODEL_VERSION)
    private String version;

    @SerializedName(DEVICE)
    private String deviceName;

    @SerializedName(ENGINE_NAME)
    private String engineName;

    @SerializedName(BATCH_SIZE)
    private int batchSize = -1;

    @SerializedName(JOB_QUEUE_SIZE)
    private int jobQueueSize = -1;

    @SerializedName(MAX_IDLE_TIME)
    private int maxIdleSeconds = -1;

    @SerializedName(MAX_BATCH_DELAY)
    private int maxBatchDelayMillis = -1;

    @SerializedName(MIN_WORKER)
    private int minWorkers = -1;

    @SerializedName(MAX_WORKER)
    private int maxWorkers = -1;

    @SerializedName(SYNCHRONOUS)
    private boolean synchronous = true;

    public LoadModelRequest() {}

    public LoadModelRequest(QueryStringDecoder decoder) {
        modelUrl = NettyUtils.getParameter(decoder, URL, null);
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }
        modelName = NettyUtils.getParameter(decoder, MODEL_NAME, null);
        if (modelName == null || modelName.isEmpty()) {
            modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
        }
        version = NettyUtils.getParameter(decoder, MODEL_VERSION, null);
        deviceName = NettyUtils.getParameter(decoder, DEVICE, null);
        engineName = NettyUtils.getParameter(decoder, ENGINE_NAME, null);
        jobQueueSize = NettyUtils.getIntParameter(decoder, JOB_QUEUE_SIZE, -1);
        batchSize = NettyUtils.getIntParameter(decoder, BATCH_SIZE, -1);
        maxBatchDelayMillis = NettyUtils.getIntParameter(decoder, MAX_BATCH_DELAY, -1);
        maxIdleSeconds = NettyUtils.getIntParameter(decoder, MAX_IDLE_TIME, -1);
        minWorkers = NettyUtils.getIntParameter(decoder, MIN_WORKER, -1);
        maxWorkers = NettyUtils.getIntParameter(decoder, MAX_WORKER, -1);
        synchronous = Boolean.parseBoolean(NettyUtils.getParameter(decoder, SYNCHRONOUS, "true"));
    }

    public String getModelUrl() {
        return modelUrl;
    }

    public String getModelName() {
        return modelName;
    }

    public String getVersion() {
        return version;
    }

    public String getDeviceName() {
        return deviceName;
    }

    public String getEngineName() {
        return engineName;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getJobQueueSize() {
        return jobQueueSize;
    }

    public int getMaxIdleSeconds() {
        return maxIdleSeconds;
    }

    public int getMaxBatchDelayMillis() {
        return maxBatchDelayMillis;
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public boolean isSynchronous() {
        return synchronous;
    }
}
