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
package ai.djl.tritonserver.engine;

import ai.djl.engine.EngineException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.Utils;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.tritonserver.global.tritonserver;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Error;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequestReleaseFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponse;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponseCompleteFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Message;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocator;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocatorAllocFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocatorReleaseFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ServerOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;

/** A class containing utilities to interact with the TritonServer C API. */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private static final Map<Pointer, CompletableFuture<TRITONSERVER_InferenceResponse>> RESPONSES =
            new ConcurrentHashMap<>();
    private static final ResponseAlloc RESP_ALLOC = new ResponseAlloc();
    private static final ResponseRelease RESP_RELEASE = new ResponseRelease();
    private static final InferRequestComplete REQUEST_COMPLETE = new InferRequestComplete();
    private static final InferResponseComplete RESPONSE_COMPLETE = new InferResponseComplete();

    private JniUtils() {}

    public static TRITONSERVER_Server initTritonServer() {
        // TODO: Use TRITONSERVER_ServerRegisterModelRepository after triton engine initialized
        String modelStore = Utils.getEnvOrSystemProperty("SERVING_MODEL_STORE");
        if (modelStore == null || modelStore.isEmpty()) {
            modelStore = "/opt/ml/model";
        }
        int[] major = {0};
        int[] minor = {0};
        checkCall(tritonserver.TRITONSERVER_ApiVersion(major, minor));
        if ((tritonserver.TRITONSERVER_API_VERSION_MAJOR != major[0])
                || (tritonserver.TRITONSERVER_API_VERSION_MINOR > minor[0])) {
            throw new EngineException("triton API version mismatch");
        }

        TRITONSERVER_ServerOptions options = new TRITONSERVER_ServerOptions(null);
        checkCall(tritonserver.TRITONSERVER_ServerOptionsNew(options));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetModelRepositoryPath(options, modelStore));
        String verboseLevel = Utils.getEnvOrSystemProperty("TRITON_VERBOSE_LEVEL");
        int verbosity;
        if (verboseLevel != null) {
            verbosity = Integer.parseInt(verboseLevel);
        } else {
            verbosity = logger.isTraceEnabled() ? 1 : 0;
        }
        checkCall(tritonserver.TRITONSERVER_ServerOptionsSetLogVerbose(options, verbosity));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetBackendDirectory(
                        options, "/opt/tritonserver/backends"));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
                        options, "/opt/tritonserver/repoagents"));
        checkCall(tritonserver.TRITONSERVER_ServerOptionsSetStrictModelConfig(options, true));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetModelControlMode(
                        options, tritonserver.TRITONSERVER_MODEL_CONTROL_EXPLICIT));

        // TODO: Stop triton and delete triton. Currently TritonServer will live for ever
        TRITONSERVER_Server triton = new TRITONSERVER_Server(null);
        checkCall(tritonserver.TRITONSERVER_ServerNew(triton, options));
        checkCall(tritonserver.TRITONSERVER_ServerOptionsDelete(options));

        // Wait until the triton is both live and ready.
        for (int i = 0; i < 10; ++i) {
            boolean[] live = {false};
            boolean[] ready = {false};
            checkCall(tritonserver.TRITONSERVER_ServerIsLive(triton, live));
            checkCall(tritonserver.TRITONSERVER_ServerIsReady(triton, ready));
            logger.debug("Triton health: live {}, ready: {}", live[0], ready[0]);
            if (live[0] && ready[0]) {
                printServerStatus(triton);
                return triton;
            }
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                throw new EngineException("Triton startup interrupted.", e);
            }
        }
        throw new EngineException("Failed to find triton healthy.");
    }

    public static void loadModel(TRITONSERVER_Server triton, String modelName, int timeout) {
        checkCall(tritonserver.TRITONSERVER_ServerLoadModel(triton, modelName));

        // Wait for the model to become available.
        boolean[] ready = {false};
        while (!ready[0]) {
            checkCall(tritonserver.TRITONSERVER_ServerModelIsReady(triton, modelName, 1, ready));
            if (ready[0]) {
                break;
            }

            if (timeout < 0) {
                throw new EngineException("Model loading timed out in: " + timeout);
            }

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                throw new EngineException("Model loading interrupted.", e);
            }
            timeout -= 500;
        }
    }

    public static void unloadModel(TRITONSERVER_Server triton, String modelName, int timeout) {
        tritonserver.TRITONSERVER_ServerUnloadModel(triton, modelName);
        // TODO: wait model fully unloaded
    }

    public static ModelMetadata getModelMetadata(TRITONSERVER_Server triton, String modelName) {
        TRITONSERVER_Message metadata = new TRITONSERVER_Message(null);
        checkCall(tritonserver.TRITONSERVER_ServerModelMetadata(triton, modelName, 1, metadata));
        BytePointer buffer = new BytePointer((Pointer) null);
        SizeTPointer size = new SizeTPointer(1);
        checkCall(tritonserver.TRITONSERVER_MessageSerializeToJson(metadata, buffer, size));
        String json = buffer.limit(size.get()).getString();
        checkCall(tritonserver.TRITONSERVER_MessageDelete(metadata));
        return JsonUtils.GSON.fromJson(json, ModelMetadata.class);
    }

    public static Output predict(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDManager manager, Input input)
            throws TranslateException {
        TRITONSERVER_InferenceRequest req = toRequest(triton, metadata, manager, input);
        return predict(triton, metadata, manager, req);
    }

    public static NDList predict(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDManager manager, NDList input)
            throws TranslateException {
        TRITONSERVER_InferenceRequest req = toRequest(triton, metadata, input);
        Output output = predict(triton, metadata, manager, req);
        return output.getAsNDList(manager, 0);
    }

    private static Output predict(
            TRITONSERVER_Server triton,
            ModelMetadata metadata,
            NDManager manager,
            TRITONSERVER_InferenceRequest req)
            throws TranslateException {
        // TODO: Can this allocator be re-used?
        TRITONSERVER_ResponseAllocator allocator = new TRITONSERVER_ResponseAllocator(null);
        checkCall(
                tritonserver.TRITONSERVER_ResponseAllocatorNew(
                        allocator, RESP_ALLOC, RESP_RELEASE, null));

        for (DataDescriptor dd : metadata.outputs) {
            checkCall(tritonserver.TRITONSERVER_InferenceRequestAddRequestedOutput(req, dd.name));
        }

        // Perform inference...
        try {
            CompletableFuture<TRITONSERVER_InferenceResponse> future = new CompletableFuture<>();
            RESPONSES.put(req, future);

            checkCall(
                    tritonserver.TRITONSERVER_InferenceRequestSetResponseCallback(
                            req, allocator, null, RESPONSE_COMPLETE, req));

            checkCall(tritonserver.TRITONSERVER_ServerInferAsync(triton, req, null));

            // Wait for the inference to complete.
            TRITONSERVER_InferenceResponse resp = future.get();
            RESPONSES.remove(req);

            checkCall(tritonserver.TRITONSERVER_InferenceResponseError(resp));

            Output output = toOutput(resp, metadata, manager);
            checkCall(tritonserver.TRITONSERVER_InferenceResponseDelete(resp));

            return output;
        } catch (ExecutionException | InterruptedException e) {
            throw new TranslateException(e);
        } finally {
            checkCall(tritonserver.TRITONSERVER_InferenceRequestDelete(req));
            checkCall(tritonserver.TRITONSERVER_ResponseAllocatorDelete(allocator));
        }
    }

    private static TRITONSERVER_InferenceRequest toRequest(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDManager manager, Input input)
            throws TranslateException {
        if (input.getProperty("Content-Type", "").startsWith("tensor/")) {
            NDList list = input.getAsNDList(manager, 0);
            return toRequest(triton, metadata, list);
        }

        PairList<String, BytesSupplier> content = input.getContent();
        if (content.size() != metadata.inputs.length) {
            throw new TranslateException(
                    "Expect " + metadata.inputs.length + " inputs, got: " + content.size());
        }

        // TODO: test binary input case
        byte[][] buf = new byte[metadata.inputs.length][];
        boolean hasName = content.contains(metadata.inputs[0].name);
        for (int i = 0; i < buf.length; ++i) {
            DataDescriptor dd = metadata.inputs[i];
            BytesSupplier bs = hasName ? input.get(dd.name) : input.get(i);
            if (bs == null) {
                throw new TranslateException("Missing input: " + dd.name);
            }
            buf[i] = bs.getAsBytes();
            if (dd.isBinaryData()) {
                int len = dd.getBinaryDataSize();
                if (len != buf[i].length) {
                    throw new TranslateException(
                            "Expect length: "
                                    + len
                                    + ", got: "
                                    + buf[i].length
                                    + " for "
                                    + dd.name);
                }
            } else {
                // TODO: Handle mixed inputs
                throw new TranslateException("Requires NDList as input " + dd.name);
            }
        }

        return toRequest(triton, metadata, buf);
    }

    private static TRITONSERVER_InferenceRequest toRequest(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDList list)
            throws TranslateException {
        if (list.size() != metadata.inputs.length) {
            throw new TranslateException(
                    "Expect " + metadata.inputs.length + " inputs, got: " + list.size());
        }

        byte[][] buf = new byte[metadata.inputs.length][];
        boolean hasName = list.head().getName() != null;
        for (int i = 0; i < buf.length; ++i) {
            DataDescriptor dd = metadata.inputs[i];
            NDArray array = hasName ? list.get(dd.name) : list.get(i);
            if (array == null) {
                throw new TranslateException("Missing input: " + dd.name);
            }
            if (dd.shapeNotEquals(array.getShape().getShape())) {
                throw new TranslateException(
                        "Input(" + dd.name + ") shape mismatch " + array.getShape());
            }
            if (dd.dataTypeNotEquals(array.getDataType())) {
                throw new TranslateException(
                        "Input(" + dd.name + ") datatype mismatch " + array.getDataType());
            }
            buf[i] = array.toByteArray();
        }

        return toRequest(triton, metadata, buf);
    }

    private static TRITONSERVER_InferenceRequest toRequest(
            TRITONSERVER_Server triton, ModelMetadata metadata, byte[][] buf) {

        // TODO: How to clean up pointers in exception case?
        TRITONSERVER_InferenceRequest req = new TRITONSERVER_InferenceRequest(null);
        checkCall(tritonserver.TRITONSERVER_InferenceRequestNew(req, triton, metadata.name, -1));

        // TODO: REQUEST_ID
        checkCall(tritonserver.TRITONSERVER_InferenceRequestSetId(req, "my_request_id"));
        checkCall(
                tritonserver.TRITONSERVER_InferenceRequestSetReleaseCallback(
                        req, REQUEST_COMPLETE, null));

        for (int i = 0; i < buf.length; ++i) {
            DataDescriptor dd = metadata.inputs[i];
            BytePointer bp = new BytePointer(buf[i]);
            dd.shape[0] = 1;
            checkCall(
                    tritonserver.TRITONSERVER_InferenceRequestAddInput(
                            req, dd.name, dd.datatype.ordinal(), dd.shape, dd.shape.length));
            // input/output always copy to CPU for now
            checkCall(
                    tritonserver.TRITONSERVER_InferenceRequestAppendInputData(
                            req,
                            dd.name,
                            bp,
                            buf[i].length,
                            tritonserver.TRITONSERVER_MEMORY_CPU,
                            0));
        }
        return req;
    }

    private static Output toOutput(
            TRITONSERVER_InferenceResponse resp, ModelMetadata metadata, NDManager manager)
            throws TranslateException {
        int[] outputCount = {0};
        checkCall(tritonserver.TRITONSERVER_InferenceResponseOutputCount(resp, outputCount));
        if (outputCount[0] != metadata.outputs.length) {
            throw new TranslateException(
                    "Expecting " + metadata.outputs.length + " outputs, got: " + outputCount[0]);
        }

        byte[][] out = new byte[outputCount[0]][];
        Shape[] shapes = new Shape[out.length];
        for (int i = 0; i < outputCount[0]; ++i) {
            BytePointer cname = new BytePointer((Pointer) null);
            IntPointer datatype = new IntPointer(1);
            LongPointer shape = new LongPointer((Pointer) null);
            LongPointer dimCount = new LongPointer(1);
            Pointer base = new Pointer();
            SizeTPointer byteSize = new SizeTPointer(1);
            IntPointer memoryType = new IntPointer(1);
            LongPointer memoryTypeId = new LongPointer(1);
            Pointer userPtr = new Pointer();

            checkCall(
                    tritonserver.TRITONSERVER_InferenceResponseOutput(
                            resp,
                            i,
                            cname,
                            datatype,
                            shape,
                            dimCount,
                            base,
                            byteSize,
                            memoryType,
                            memoryTypeId,
                            userPtr));

            if (cname.isNull()) {
                throw new TranslateException("Unable to get output name.");
            }
            String name = cname.getString();
            DataDescriptor dd = metadata.getOutput(name);
            if (dd == null) {
                throw new TranslateException("Unexpected output name: " + name);
            }
            if (!dd.isBinaryData()) {
                int size = Math.toIntExact(dimCount.get());
                long[] s = new long[size];
                shape.get(s);
                if (dd.shapeNotEquals(s)) {
                    throw new TranslateException("Unexpected shape for " + name);
                }
                shapes[dd.index] = new Shape(s);
                if (datatype.get() != dd.datatype.ordinal()) {
                    throw new TranslateException(
                            "Unexpected datatype '"
                                    + tritonserver.TRITONSERVER_DataTypeString(datatype.get())
                                    + "' for '"
                                    + name
                                    + "'");
                }
            }
            int len = Math.toIntExact(byteSize.get());
            // make a copy here
            byte[] buf = new byte[len];
            base.limit(len).asByteBuffer().get(buf);
            out[dd.index] = buf;
        }

        Output output = new Output();
        if (metadata.isOutputAllTensor()) {
            // TODO: Respect Accept header properly
            NDList list = new NDList();
            for (int i = 0; i < out.length; ++i) {
                DataDescriptor dd = metadata.outputs[i];
                NDArray array = manager.create(ByteBuffer.wrap(out[i]), shapes[i]);
                array.setName(dd.name);
                list.add(array);
            }

            output.addProperty("Content-Type", "tensor/ndlist");
            output.add(list);
        } else {
            // TODO: how to encode tensor data in mixed output?
            for (int i = 0; i < out.length; ++i) {
                output.add(metadata.outputs[i].name, out[i]);
            }
        }

        return output;
    }

    private static void printServerStatus(TRITONSERVER_Server triton) {
        // Print status of the triton.
        TRITONSERVER_Message metadata = new TRITONSERVER_Message(null);
        checkCall(tritonserver.TRITONSERVER_ServerMetadata(triton, metadata));
        BytePointer buffer = new BytePointer((Pointer) null);
        SizeTPointer size = new SizeTPointer(1);
        checkCall(tritonserver.TRITONSERVER_MessageSerializeToJson(metadata, buffer, size));

        logger.info("Server Status: {}", buffer.limit(size.get()).getString());
        checkCall(tritonserver.TRITONSERVER_MessageDelete(metadata));
    }

    private static void checkCall(TRITONSERVER_Error err) {
        if (err != null) {
            String error =
                    tritonserver.TRITONSERVER_ErrorCodeString(err)
                            + " - "
                            + tritonserver.TRITONSERVER_ErrorMessage(err);
            tritonserver.TRITONSERVER_ErrorDelete(err);
            throw new EngineException(error);
        }
    }

    private static final class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("rawtypes")
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                String tensorName,
                long byteSize,
                int preferredMemoryType,
                long preferredMemoryTypeId,
                Pointer userPtr,
                PointerPointer buffer,
                PointerPointer bufferUserPtr,
                IntPointer actualMemoryType,
                LongPointer actualMemoryTypeId) {
            // Initially attempt to make the actual memory type and id that we
            // allocate be the same as preferred memory type
            actualMemoryType.put(0, preferredMemoryType);
            actualMemoryTypeId.put(0, preferredMemoryTypeId);

            // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
            // need to do any other book-keeping.
            if (byteSize == 0) {
                buffer.put(0, null);
                bufferUserPtr.put(0, null);
                logger.debug("allocated {}} bytes for result tensor {}.", byteSize, tensorName);
            } else {
                actualMemoryType.put(0, tritonserver.TRITONSERVER_MEMORY_CPU);
                Pointer allocatedPtr = Pointer.malloc(byteSize);
                // Pass the tensor name with bufferUserPtr so we can show it when
                // releasing the buffer.
                if (!allocatedPtr.isNull()) {
                    buffer.put(0, allocatedPtr);
                    bufferUserPtr.put(0, Loader.newGlobalRef(tensorName));
                    logger.debug(
                            "allocated {} bytes in {} for result tensor {}",
                            byteSize,
                            tritonserver.TRITONSERVER_MemoryTypeString(actualMemoryType.get()),
                            tensorName);
                } else {
                    throw new EngineException("Out of memory, malloc failed.");
                }
            }

            return null;
        }
    }

    private static final class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {

        /** {@inheritDoc} */
        @Override
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                Pointer buffer,
                Pointer bufferUserPtr,
                long byteSize,
                int memoryType,
                long memoryTypeId) {
            String name;
            if (bufferUserPtr != null) {
                name = (String) Loader.accessGlobalRef(bufferUserPtr);
            } else {
                name = "<unknown>";
            }

            logger.debug(
                    "Releasing buffer {} of size {} in {} for result '{}'",
                    buffer,
                    byteSize,
                    tritonserver.TRITONSERVER_MemoryTypeString(memoryType),
                    name);
            Pointer.free(buffer);
            Loader.deleteGlobalRef(bufferUserPtr);

            return null; // Success
        }
    }

    private static final class InferRequestComplete
            extends TRITONSERVER_InferenceRequestReleaseFn_t {

        /** {@inheritDoc} */
        @Override
        public void call(TRITONSERVER_InferenceRequest request, int flags, Pointer userp) {
            // We reuse the request so we don't delete it here.
        }
    }

    private static final class InferResponseComplete
            extends TRITONSERVER_InferenceResponseCompleteFn_t {

        /** {@inheritDoc} */
        @Override
        public void call(TRITONSERVER_InferenceResponse response, int flags, Pointer userp) {
            if (response != null) {
                // Send 'response' to the future.
                RESPONSES.get(userp).complete(response);
            }
        }
    }

    static final class DataDescriptor {

        String name;
        TsDataType datatype;
        long[] shape;
        Map<String, Object> parameters;
        int index;

        public void setName(String name) {
            this.name = name;
        }

        public void setDatatype(TsDataType datatype) {
            this.datatype = datatype;
        }

        public void setShape(long[] shape) {
            this.shape = shape;
        }

        public void setParameters(Map<String, Object> parameters) {
            this.parameters = parameters;
        }

        public boolean isBinaryData() {
            if (parameters == null) {
                return false;
            }
            return ArgumentsUtil.booleanValue(parameters, "binary_data");
        }

        public int getBinaryDataSize() {
            if (parameters == null) {
                return -1;
            }
            return ArgumentsUtil.intValue(parameters, "binary_data_size", -1);
        }

        public boolean dataTypeNotEquals(DataType other) {
            return datatype.toDataType() != other;
        }

        public boolean shapeNotEquals(long[] other) {
            if (shape.length != other.length) {
                return true;
            }
            for (int i = 0; i < shape.length; ++i) {
                if (shape[i] != other[i] && shape[i] != -1 && other[i] != -1) {
                    return true;
                }
            }
            return false;
        }
    }

    static final class ModelMetadata {

        String name;
        String[] versions;
        String platform;
        DataDescriptor[] inputs;
        DataDescriptor[] outputs;

        public void setName(String name) {
            this.name = name;
        }

        public void setVersions(String[] versions) {
            this.versions = versions;
        }

        public void setPlatform(String platform) {
            this.platform = platform;
        }

        public void setInputs(DataDescriptor[] inputs) {
            this.inputs = inputs;
        }

        public void setOutputs(DataDescriptor[] outputs) {
            this.outputs = outputs;
        }

        public DataDescriptor getOutput(String name) {
            int index = 0;
            for (DataDescriptor dd : outputs) {
                if (dd.name.equals(name)) {
                    dd.index = index;
                    return dd;
                }
                ++index;
            }
            return null;
        }

        public boolean isInputAllTensors() {
            for (DataDescriptor dd : inputs) {
                if (dd.isBinaryData()) {
                    return false;
                }
            }
            return true;
        }

        public boolean isOutputAllTensor() {
            for (DataDescriptor dd : outputs) {
                if (dd.isBinaryData()) {
                    return false;
                }
            }
            return true;
        }
    }
}
