package ai.djl.serving.plugins.securemode;

import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.ModelServerListenerAdapter;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    private static final Logger LOGGER = LoggerFactory.getLogger(SecureModeModelServerListener.class);

    private static void foo(ModelInfo<?, ?> model) {
        LOGGER.info("Resolving Draft Model for the Model Configuration...");
    }

    @Override
    public void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath) {
        super.onModelDownloaded(model, downloadPath);
        // do stuff
    }
}
