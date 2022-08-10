<template>
  <div class="log">
    <h3>Log</h3>
    <div class="log-box">
      <div class="title">Running Log</div>
      <div class="log-content">
        <pre>
        [INFO ] - Starting djl-serving: 0.18.0 ...
        [INFO ] -
        Model server home: D:\deeplearning\djl-serving-master
        Current directory: D:\deeplearning\djl-serving-master
        Temp directory: C:\Users\Administrator\AppData\Local\Temp\
        Number of CPUs: 4
        Max heap size: 2267
        Config file: D:\serving-0.17.0\conf\config.properties
        Inference address: http://127.0.0.1:8080
        Management address: http://127.0.0.1:8080
        Model Store: D:\deeplearning\djl-serving-master\models
        Initial Models: file:///D:/serving-0.17.0/build/models/resnet18_v1,[resnet:v1:MXNet:*]=file:///D:/serving-0.17.0/build/models/resnet18_v1
        Initial Workflows: N/A
        Netty threads: 0
        Maximum Request Size: 6553500
        [INFO ] - Initializing model: file:///D:/serving-0.17.0/build/models/resnet18_v1
        [INFO ] - Loading model resnet18_v1 on cpu().
        [08:14:10] C:\source\mxnet\src\nnvm\legacy_json_util.cc:208: Loading symbol saved by previous version v1.5.0. Attempting to upgrade...
        [08:14:10] C:\source\mxnet\src\nnvm\legacy_json_util.cc:216: Symbol successfully upgraded!
        [INFO ] - Initializing model: [resnet:v1:MXNet:*]=file:///D:/serving-0.17.0/build/models/resnet18_v1
        [INFO ] - Loading model resnet on cpu().
        [08:14:11] C:\source\mxnet\src\nnvm\legacy_json_util.cc:208: Loading symbol saved by previous version v1.5.0. Attempting to upgrade...
        [08:14:11] C:\source\mxnet\src\nnvm\legacy_json_util.cc:216: Symbol successfully upgraded!
        [INFO ] - scanning for plugins...
        [INFO ] - 0 plug-ins found and loaded.
        [INFO ] - Initialize BOTH server with: NioServerSocketChannel.
        [INFO ] - BOTH API bind to: http://127.0.0.1:8080
        [INFO ] - Model server started.
         </pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Log",
  components: {

  },
  props: {

  },
  data() {
    return {

    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  mounted() {

  },
  methods: {

  },
};
</script>

<style  lang="less">
.log {
  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin-top: 0px;
  }
  .log-box {
    background: #fff;
  }
  .title {
    background: #e5ffee;
    height: 40px;
    display: flex;
    align-items: center;
    font-size: @titleSize4;
    // padding-bottom: 20px;
    padding-left: 40px;
    position: relative;
    &::before {
      position: absolute;
      content: "";
      width: 10px;
      height: 10px;
      background: #02f21a;
      border-radius: 10px;
      display: block;
      top: 16px;
      left: 15px;
    }
  }
}
</style>
