<template>
  <div class="log">
    <h3>Log Info</h3>
    <div class="log-box">
      <div class="title">{{logName}}</div>
      <div class="log-content" ref="logContent">
        <pre>{{logText}}</pre>
      </div>
    </div>
  </div>
</template>

<script>
import * as logApi from "@/api/logAPI"

export default {
  name: "LogInfo",
  components: {

  },
  props: {

  },
  data() {
    return {
      logName: "",
      logText: ""
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    this.logName = this.$route.params.name
    let res = await logApi.logInfo(this.logName)
    this.logText = res
    this.$nextTick(() => {
      let middle = this.$refs["logContent"];
      middle.scrollTop = middle.scrollHeight;
    })
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
    .log-content {
      height: calc(100vh - 200px);
      overflow: auto;
    }
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
  pre {
    padding: 0 20px;
    white-space: pre-wrap;
  }
}
</style>
