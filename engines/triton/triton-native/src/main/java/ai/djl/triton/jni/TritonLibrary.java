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
package ai.djl.triton.jni;

import java.nio.ByteBuffer;

public final class TritonLibrary {

    public static final TritonLibrary LIB = new TritonLibrary();

    private TritonLibrary() {}

    public native long createServerOption(String repositoryPath, String repoBackendsPath, String repoAgentPath, int verbose);

    public native void deleteServerOption(long option);

    public native long createModelServer(long option);

    public native void deleteModelServer(long serverHandle);

    public native long buildInferenceRequest(long serverHandle, String modelName, String id);

    public native void deleteInferenceRequest(long irHandle);

    public native void addInput(long irHandle, String name, ByteBuffer data, long byteSize, long[] shape, int dtype);

    public native void addOutput(long irHandle, String name);

    public native long[] performInference(long serverHandle, long irHandle);

    public native ByteBuffer[] fetchResult(long responseHandle, long[][] shape, int[] dtype);

    public native void deleteResponse(long[] handles);

}
