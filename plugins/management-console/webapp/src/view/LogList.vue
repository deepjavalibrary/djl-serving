<template>
  <div class="log-list">
    <h3>Log List</h3>
    <div class="log-box">
      <div class="title">Running Log</div>
      <div class="log-content">
        <el-table :data="logList" stripe style="width: 100%">
          <el-table-column prop="name" label="Name">
          </el-table-column>
          <el-table-column prop="length" label="Length">
            <template slot-scope="scope">
              {{scope.row.length|byteConvert}}
            </template>
          </el-table-column>
          <el-table-column prop="lastModified" label="Last Modified">
            <template slot-scope="scope">
              {{scope.row.lastModified|dateFormat}}
            </template>
          </el-table-column>
          <el-table-column label="Operation">
            <template slot-scope="scope">
              <el-button @click="detail(scope.row)" type="text" size="small">Details</el-button>
              <el-button @click="download(scope.row)" type="text" size="small">Download</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script>
import * as logApi from "@/api/logAPI"

export default {
  name: "LogList",
  components: {

  },
  props: {

  },
  data() {
    return {
      logList: []
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    let res = await logApi.logs()
    this.logList = res
  },
  methods: {
    detail(val) {
      this.$router.push("/log/" + val.name)
    },
    download(val) {
      logApi.download(val.name).then(res => {
        // console.log(res);
        const link = document.createElement('a');
        let blob = new Blob([res]);
        link.style.display = 'none';
        link.href = URL.createObjectURL(blob);

        link.setAttribute('download', val.name);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link)
      })

    }
  },
};
</script>

<style  lang="less">
.log-list {
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
  .log-content {
    padding: 20px;
  }
}
</style>
