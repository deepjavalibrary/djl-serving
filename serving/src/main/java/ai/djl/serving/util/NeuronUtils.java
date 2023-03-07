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

    private NeuronUtils() {}

    /**
     * Gets whether Neuron runtime library is in the system.
     *
     * @return {@code true} if Neuron runtime library is in the system
     */
    public static boolean hasNeuron() {
        return getNeuronCores() > 0;
    }

    /**
     * Checks if inf2 is used.
     *
     * @return is inf2
     */
    public static boolean isInf2() {
        String metadata = Ec2Utils.readMetadata("instance-type");
        if (metadata == null) {
            return false;
        }
        return metadata.startsWith("inf2") || metadata.startsWith("trn1");
    }

    /**
     * Checks if inf1 is used.
     *
     * @return is inf1
     */
    public static boolean isInf1() {
        String metadata = Ec2Utils.readMetadata("instance-type");
        if (metadata == null) {
            return false;
        }
        return metadata.startsWith("inf1");
    }

    /**
     * Returns the number of NeuronCores available in the system.
     *
     * @return the number of NeuronCores available in the system
     */
    public static int getNeuronCores() {
        if (isInf1() || isInf2()) {
            try (Stream<Path> paths = Files.walk(Paths.get("/dev"))) {
                int nd = (int) paths.filter(ele -> ele.startsWith("neuron")).count();
                if (isInf1()) {
                    return nd * 4;
                } else if (isInf2()) {
                    return nd * 2;
                } else {
                    return 0;
                }
            } catch (IOException ignore) {
                return 0;
            }
        }
        return 0;
    }
}
