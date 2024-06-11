package ai.djl.serving.plugins.securemode;

import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.ModelServerListenerAdapter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SecureModeModelServerListener.class);

    private static void foo(ModelInfo<?, ?> model) {
        LOGGER.info("Resolving Draft Model for the Model Configuration...");
    }

    @Override
    public void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath) {
        super.onModelDownloaded(model, downloadPath);
        LOGGER.info("MODEL PROPERTIES: {}", model.getProperties());
        LOGGER.info("MODEL URL: {}", model.getModelUrl());
    }
}
