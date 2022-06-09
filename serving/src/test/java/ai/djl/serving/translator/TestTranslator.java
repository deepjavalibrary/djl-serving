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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

public class TestTranslator implements ServingTranslator {

    private int topK = 10;
    private List<String> classes;

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        classes = ctx.getModel().getArtifact("synset.txt", Utils::readLines);
    }

    @Override
    public void setArguments(java.util.Map<String, ?> arguments) {
        topK = ArgumentsUtil.intValue(arguments, "topK", topK);
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilitiesNd = list.singletonOrThrow();
        NDArray max = probabilitiesNd.argMax();
        Output output = new Output();
        output.add("topK: " + topK + ", best: " + classes.get((int) max.getLong()));
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
}
