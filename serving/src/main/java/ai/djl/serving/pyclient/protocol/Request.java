package ai.djl.serving.pyclient.protocol;

/**
 * Request format to python server.
 * TODO: Will be changed to support python file, method, process function type.
 */
public class Request {
    private byte[] rawData;

    /**
     * Sets the request data
     *
     * @param rawData request data in bytes
     */
    public Request(byte[] rawData) {
        this.rawData = rawData;
    }

    /**
     * Getter for rawData
     *
     * @return rawData
     */
    public byte[] getRawData() {
        return rawData;
    }

    /**
     * Setter for rawData
     *
     * @param rawData request data in bytes
     */
    public void setRawData(byte[] rawData) {
        this.rawData = rawData;
    }
}
