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

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

class PyPredictor<I, O> extends Predictor<I, O> {

    private PyProcess process;

    public PyPredictor(Model model, Translator<I, O> translator, PyEnv pyEnv) {
        super(model, translator, false);
        process = new PyProcess(model, pyEnv);
        process.startPythonProcess();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public O predict(I input) throws TranslateException {
        if (process.isStopped()) {
            throw new TranslateException("Backend Python process is stopped.");
        }
        if (input instanceof Input) {
            return (O) process.predict((Input) input);
        }
        return super.predict(input);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList)
            throws TranslateException {
        Input inputs = new Input(null);
        inputs.addData(ndList.encode());
        Output output = process.predict(inputs);
        if (output.getCode() != 200) {
            throw new TranslateException(output.getMessage());
        }
        NDManager manager = ndList.head().getManager();
        return NDList.decode(manager, output.getContent());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        process.stopPythonProcess();
    }
}
