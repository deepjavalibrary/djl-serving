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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;

class PyPredictor<I, O> extends Predictor<I, O> {

    private PyProcess process;
    private int timeout;

    public PyPredictor(Model model, Translator<I, O> translator, PyEnv pyEnv, Device device) {
        super(model, translator, device, false);
        timeout = pyEnv.getPredictTimeout();
        process = new PyProcess(model, pyEnv);
        process.startPythonProcess();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        if (process.isStopped()) {
            throw new TranslateException("Backend Python process is stopped.");
        }
        if (inputs.get(0) instanceof Input) {
            List<O> ret = new ArrayList<>(inputs.size());
            for (I input : inputs) {
                ret.add((O) process.predict((Input) input, timeout, false));
            }
            return ret;
        }
        return super.batchPredict(inputs);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList)
            throws TranslateException {
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
    }
}
