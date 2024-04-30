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
package ai.djl.benchmark;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDList.Encoding;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import ai.djl.util.passthrough.PassthroughNDManager;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** A class generates NDList files. */
final class NDListGenerator {

    private static final Logger logger = LoggerFactory.getLogger(NDListGenerator.class);

    private NDListGenerator() {}

    static boolean generate(String[] args) {
        Options options = getOptions();
        try {
            if (Arguments.hasHelp(args)) {
                Arguments.printHelp(
                        "usage: djl-bench ndlist-gen -s INPUT-SHAPES -o OUTPUT_FILE", options);
                return true;
            }
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            String inputShapes = cmd.getOptionValue("input-shapes");
            String output = cmd.getOptionValue("output-file");
            boolean ones = cmd.hasOption("ones");
            Encoding encoding;
            if (cmd.hasOption("npz")) {
                encoding = Encoding.NPZ;
            } else if (cmd.hasOption("safetensors")) {
                encoding = Encoding.SAFETENSORS;
            } else {
                encoding = Encoding.ND_LIST;
            }
            Path path = Paths.get(output);
            NDManager manager = PassthroughNDManager.INSTANCE;
            NDList list = new NDList();
            for (Pair<DataType, Shape> pair : Shape.parseShapes(inputShapes)) {
                DataType dataType = pair.getKey();
                Shape shape = pair.getValue();
                if (ones) {
                    list.add(manager.ones(shape, dataType));
                } else {
                    list.add(manager.zeros(shape, dataType));
                }
            }
            try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(path))) {
                list.encode(os, encoding);
            }
            logger.info("NDList file created: {}", path.toAbsolutePath());
            return true;
        } catch (ParseException e) {
            Arguments.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
        return false;
    }

    private static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
        options.addOption(
                Option.builder("s")
                        .required()
                        .longOpt("input-shapes")
                        .hasArg()
                        .argName("INPUT-SHAPES")
                        .desc("Input data shapes for the model.")
                        .build());
        options.addOption(
                Option.builder("o")
                        .required()
                        .longOpt("output-file")
                        .hasArg()
                        .argName("OUTPUT-FILE")
                        .desc("Write output NDList to file.")
                        .build());
        options.addOption(
                Option.builder("1")
                        .longOpt("ones")
                        .hasArg(false)
                        .argName("ones")
                        .desc("Use all ones instead of zeros.")
                        .build());
        OptionGroup group = new OptionGroup();
        group.addOption(
                Option.builder("z")
                        .longOpt("npz")
                        .hasArg(false)
                        .argName("npz")
                        .desc("Output .npz format.")
                        .build());
        group.addOption(
                Option.builder("st")
                        .longOpt("safetensors")
                        .hasArg(false)
                        .argName("safetensors")
                        .desc("Output .safetensors format.")
                        .build());
        options.addOptionGroup(group);
        return options;
    }
}
