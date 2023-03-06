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
     * Returns the number of NeuronCores available in the system.
     *
     * @return the number of NeuronCores available in the system
     */
    public static int getNeuronCores() {
        String metadata = Ec2Utils.readMetadata("instance-type");
        if (metadata == null) {
            return 0;
        }
        switch (metadata) {
            case "inf1.xlarge":
            case "inf1.2xlarge":
                return 4;
            case "inf1.6xlarge":
                return 16;
            case "inf1.24xlarge":
                return 64;
            case "inf2.xlarge":
            case "inf2.8xlarge":
            case "trn1.2xlarge":
                return 2;
            case "inf2.24xlarge":
                return 12;
            case "inf2.48xlarge":
                return 24;
            case "trn1.32xlarge":
                return 32;
            default:
                return 0;
        }
    }
}
