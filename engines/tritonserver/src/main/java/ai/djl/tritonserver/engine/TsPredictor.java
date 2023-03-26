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

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;

import java.util.Collections;
import java.util.List;

class TsPredictor<I, O> extends Predictor<I, O> {

    private TRITONSERVER_Server triton;
    private JniUtils.ModelMetadata metadata;

    public TsPredictor(
            TRITONSERVER_Server triton, TsModel model, Translator<I, O> translator, Device device) {
        super(model, translator, device, false);
        this.triton = triton;
        this.metadata = model.metadata;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        Object first = inputs.get(0);
        if (!(first instanceof Input)) {
            return super.batchPredict(inputs);
        }

        int size = inputs.size();
        if (size == 1) {
            Output output = JniUtils.predict(triton, metadata, manager, (Input) first);
            return Collections.singletonList((O) output);
        }

        // TODO: Adds dynamic batching support
        throw new TranslateException("Batch is supported yet.");
    }

    /** {@inheritDoc} */
    @Override
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList)
            throws TranslateException {
        return JniUtils.predict(triton, metadata, manager, ndList);
    }
}
