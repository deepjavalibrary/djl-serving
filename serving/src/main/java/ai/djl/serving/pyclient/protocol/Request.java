package ai.djl.serving.pyclient.protocol;

public class Request {
    private byte[] rawData;

    public Request(byte[] rawData) {
        this.rawData = rawData;
    }

    public byte[] getRawData() {
        return rawData;
    }

    public void setRawData(byte[] rawData) {
        this.rawData = rawData;
    }
}
