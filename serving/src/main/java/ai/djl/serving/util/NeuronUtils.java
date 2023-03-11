/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.util;

import ai.djl.util.Ec2Utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

/** A utility class to detect number of nueron cores. */
public final class NeuronUtils {

    private static int instanceType = -1;

    private NeuronUtils() {}

    static void setInstanceType(int type) {
        instanceType = type;
    }

    /**
     * Returns whether Neuron runtime library is in the system.
     *
     * @return {@code true} if Neuron runtime library is in the system
     */
    public static boolean hasNeuron() {
        return isInf1() || isInf2();
    }

    /**
     * Returns whether the instance is an inf1.
     *
     * @return {code true} if the instance is an inf1
     */
    public static boolean isInf1() {
        return getInstanceType() == 1;
    }

    /**
     * Returns whether the instance is an inf2 or trn1.
     *
     * @return {code true} if the instance is an inf2 or trn1
     */
    public static boolean isInf2() {
        return getInstanceType() == 2;
    }

    /**
     * Returns the number of NeuronCores available in the system.
     *
     * @return the number of NeuronCores available in the system
     */
    public static int getNeuronCores() {
        if (!hasNeuron()) {
            return 0;
        }
        try (Stream<Path> paths = Files.list(Paths.get("/dev"))) {
            long nd = paths.filter(p -> p.getFileName().toString().startsWith("neuron")).count();
            if (isInf1()) {
                // inf1 has 4 cores on each device
                return (int) nd * 4;
            }
            // inf2 has 2 cores on each device
            return (int) nd * 2;
        } catch (IOException e) {
            throw new AssertionError("Failed to list neuron cores", e);
        }
    }

    @SuppressWarnings("PMD.NonThreadSafeSingleton")
    private static int getInstanceType() {
        if (instanceType == -1) {
            String metadata = Ec2Utils.readMetadata("instance-type");
            if (metadata == null) {
                NeuronUtils.setInstanceType(0);
            } else if (metadata.startsWith("inf1")) {
                NeuronUtils.setInstanceType(1);
            } else if (metadata.startsWith("inf2") || metadata.startsWith("trn1")) {
                NeuronUtils.setInstanceType(2);
            } else {
                NeuronUtils.setInstanceType(0);
            }
        }
        return instanceType;
    }
}
