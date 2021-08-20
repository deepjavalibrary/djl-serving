package ai.djl.serving;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.modality.Input;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.serving.pyclient.PythonTranslator;
import ai.djl.translate.TranslatorContext;
import org.testng.annotations.Test;

public class PythonTranslatorTest {

    @Test
    public void testPythonTranslator() {
        NDManager manager = NDManager.newBaseManager();
        NDArray ndArray = manager.zeros(new Shape(2,2));
        NDList ndList = new NDList(ndArray);
        Input input = new Input("1");
        input.addData(ndList.encode());

        PythonTranslator pythonTranslator = new PythonTranslator();
        TranslatorContext context = new TranslatorContext() {
            @Override
            public Model getModel() {
                return null;
            }

            @Override
            public NDManager getNDManager() {
                return manager;
            }

            @Override
            public Metrics getMetrics() {
                return null;
            }

            @Override
            public Object getAttachment(String key) {
                return null;
            }

            @Override
            public void setAttachment(String key, Object value) {

            }

            @Override
            public void close() {

            }
        };

        try {
            NDList list = pythonTranslator.processInput(context, input);
            System.out.println(list);

        } catch (Exception exception) {
            exception.printStackTrace();
        }
    }

}
