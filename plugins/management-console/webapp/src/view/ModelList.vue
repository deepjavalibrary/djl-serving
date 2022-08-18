<template>
  <div class="model-list">
    <h3>Model List</h3>
    <div class="add-model-btn" @click="addModel">
      <i class="el-icon-circle-plus"></i>
      <span>Add Model</span>
    </div>

    <ul class="models">
      <li class="model" v-for="(model,index) in  models " :key="index">
        <div class="bg">
          <div class="model-name">
            {{model.modelName}}
          </div>
        </div>
        <div class="status-bar">
          <div :class="['status',model.status.toLocaleLowerCase()]">{{model.status}}</div>
          <div class="version" v-if="model.version"><i class="icon-ver"></i>{{model.version}}</div>
        </div>
        <div class="opt-bar">
          <i class="icon-play" @click="inference(model)"></i>
          <i class="icon-del" @click="del(model)"></i>
          <i class="icon-setting" @click="setting(model)"></i>
        </div>
      </li>

    </ul>
  </div>
</template>

<script>
import * as modelApi from "@/api/modelAPI"
export default {
  name: "ModelList",
  components: {

  },
  props: {

  },
  data() {
    return {
      activeName: 'all',
      models: []
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    await this.qryModels()
  },
  methods: {
    addModel() {
      // console.log("addmodel");
      this.$router.push("add-model")
    },
    setting(model) {
      this.$router.push("/update-model/" + model.modelName + ":" + (model.version || ""))
    },
    inference(model) {
      console.log("inference", model);
      this.$router.push("/inference/" + model.modelName + ":" + (model.version || ""))
    },
    async del(model) {
      const confirmResult = await this.$confirm('Are you sure to delete ' + name, 'Warning', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel',
        type: 'warning',
      }).catch((err) => err)
      if (confirmResult == 'confirm') {
        let res = await modelApi.delModel(model.modelName, model.version)
        this.$message.success(res.status)
        await this.qryModels()
      }
    },
    async qryModels() {
      let res = await modelApi.models()
      console.log(res);
      this.models = res.models
    }
  },
};
</script>

<style  lang="less">
.model-list {
  // padding: 0px 80px;
  // width: 1200px;
  margin: 0 auto;
  position: relative;

  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin: 0px;
  }
  .add-model-btn {
    z-index: 200;
    display: flex;
    align-items: center;
    cursor: pointer;
    justify-content: flex-end;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 2px solid #e4e7ed;
    i {
      color: @themeColor;
      font-size: 30px;
      margin-right: 10px;
    }
    span {
      font-size: @textSize;
    }
  }
  .el-tabs {
    .el-tabs__nav {
      .el-tabs__item {
        font-size: @titleSize3;
      }
      .el-tabs__item.is-active {
        color: @themeColor;
      }
    }
  }
  .models {
    display: flex;
    align-content: flex-start;
    padding: 0;
    margin: 0;
    flex-wrap: wrap;
    column-gap: 20px;
    .model {
      height: 270px;
      width: 386px;
      background: #fff;
      border: 2px solid #f5f5f5;
      border-radius: 10px;
      margin-bottom: 20px;
      // margin-right: 20px;
      
      .bg {
        height: 140px;
        background: url("../assets/img/model-bg.png") no-repeat;
        background-size: cover;
        border-radius: 10px;
        margin: 12px;
        display: flex;
        justify-content: center;
        align-items: center;
        .model-name {
          font-size: 24px;
          font-family: Arial;
          font-weight: bold;
          color: #fff;
          text-shadow: 0px 4px 0px rgba(0, 111, 162, 0.29);
          letter-spacing: 2px;
          word-break: break-all;
          text-align: center;
        }
      }
      .status-bar {
        display: flex;
        justify-content: space-between;
        margin: 0px 20px;
        border-bottom: 2px solid #f5f5f5;
        padding-bottom: 10px;
        .status {
          font-size: 16px;
          color: #44fd75;
          padding-left: 20px;
          position: relative;
          &::before {
            content: "";
            display: block;
            width: 10px;
            height: 10px;
            background: #44fd75;
            border-radius: 10px;
            position: absolute;
            left: 0px;
            top: 6px;
            box-shadow: 0px 0px 3px rgb(0 255 129);
          }
        }
        .status.failed {
          color: #fd444e;
          &::before {
            background: #fd444e;
            box-shadow: 0px 0px 3px #fd444e;
          }
        }
        .status.running {
          color: #ffa500;
          &::before {
            background: #ffa500;
            box-shadow: 0px 0px 3px #ffa500;
          }
        }
        .version {
          display: flex;
          align-items: center;
          font-size: @textSize;
          color: #9391a5;
          i {
            margin-right: 5px;
          }
        }
      }
      .opt-bar {
        font-size: 24px;
        color: #818392;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        i {
          cursor: pointer;
        }
      }
    }
  }
  // .model:nth-child(3n) {
  //   margin-right: 0;
  // }

}
</style>
