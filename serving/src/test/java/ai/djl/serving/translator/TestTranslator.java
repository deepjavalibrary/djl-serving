/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.translator;

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import org.testng.collections.Sets;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.Set;

public class TestTranslator implements ServingTranslator, TranslatorFactory {

    private float threshold = 0.2f;

    @Override
    public void setArguments(java.util.Map<String, ?> arguments) {
        this.threshold = (float) arguments.get("threshold");
    }


    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilitiesNd = list.singletonOrThrow();
        probabilitiesNd = probabilitiesNd.softmax(0);
        Output output = new Output();
        output.add("Test succeeded " + threshold);
        return output;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        String url = input.getAsString(0);
        Image img = ImageFactory.getInstance().fromUrl(url);
        NDArray arr = img.toNDArray(ctx.getNDManager());
        arr = NDImageUtils.resize(arr, 224);
        arr = NDImageUtils.toTensor(arr);
        return new NDList(arr);
    }

    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Sets.newHashSet(new Pair<>(Input.class, Output.class));
    }

    @Override
    public Translator<?, ?> newInstance(Class<?> input, Class<?> output, Model model, Map<String, ?> arguments) {
        this.setArguments(arguments);
        return this;
    }
}
