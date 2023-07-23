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
package ai.djl.python.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class PyPredictor<I, O> extends Predictor<I, O> {

    private static final Pattern BATCH_PATTERN = Pattern.compile("batch_(\\d+)\\.(.*)");

    private PyProcess process;
    private int timeout;
    private boolean isRollingBatch;
    private RollingBatch rollingBatch;

    public PyPredictor(
            Model model,
            PyProcess process,
            int timeout,
            Translator<I, O> translator,
            Device device) {
        super(model, translator, device, false);
        this.process = process;
        this.timeout = timeout;
        isRollingBatch = model.getProperty("rolling_batch") != null;
        if (isRollingBatch) {
            String outputFormatter = model.getProperty("output_formatter");
            int maxRollingBatchSize =
                    Integer.parseInt(model.getProperty("max_rolling_batch_size", "32"));
            rollingBatch = new RollingBatch(process, maxRollingBatchSize, timeout, outputFormatter);
        }
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        if (process.isStopped()) {
            // TODO: wait for restart
            throw new TranslateException("Backend Python process is stopped.");
        }
        Object first = inputs.get(0);
        if (first instanceof Input) {
            int size = inputs.size();
            if (size == 1) {
                Output output;
                if (isRollingBatch) {
                    output = rollingBatch.addInput((Input) first, timeout);
                } else {
                    output = process.predict((Input) first, timeout, false);
                }
                return Collections.singletonList((O) output);
            }

            Input batch = new Input();
            List<O> ret = new ArrayList<>(size);
            batch.setProperties(((Input) first).getProperties());
            batch.addProperty("batch_size", String.valueOf(size));
            for (int i = 0; i < size; ++i) {
                Input in = (Input) inputs.get(i);
                PairList<String, BytesSupplier> content = in.getContent();
                String prefix = "batch_" + i;
                for (Pair<String, BytesSupplier> pair : content) {
                    String key = pair.getKey();
                    key = key == null ? "data" : key;
                    batch.add(prefix + '.' + key, pair.getValue());
                }
            }
            Output output = process.predict(batch, timeout, false);
            if (output.getCode() >= 300) {
                for (int i = 0; i < size; ++i) {
                    ret.add((O) output);
                }
                return ret;
            }
            if (output.getContent().size() != size) {
                throw new TranslateException(
                        "Batch output size mismatch, expected: "
                                + size
                                + ", actual: "
                                + output.getContent().size());
            }
            for (int i = 0; i < size; ++i) {
                Output out = new Output();
                out.setCode(output.getCode());
                out.setMessage(output.getMessage());
                out.setProperties(output.getProperties());
                ret.add((O) out);
            }

            PairList<String, BytesSupplier> content = output.getContent();
            for (Pair<String, BytesSupplier> pair : content) {
                String key = pair.getKey();
                Matcher m = BATCH_PATTERN.matcher(key);
                if (!m.matches()) {
                    throw new TranslateException("Unexpected batch output key: " + key);
                }
                int index = Integer.parseInt(m.group(1));
                Output out = (Output) ret.get(index);
                out.add(m.group(2), pair.getValue());
            }
            return ret;
        }
        return super.batchPredict(inputs);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList) {
        Input inputs = new Input();
        inputs.addProperty("Content-Type", "tensor/ndlist");
        inputs.add(ndList.encode());
        Output output = process.predict(inputs, timeout, false);
        NDManager manager = ndList.head().getManager();
        return output.getDataAsNDList(manager);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        process.stopPythonProcess();
        if (rollingBatch != null) {
            rollingBatch.shutdown();
        }
    }
}
