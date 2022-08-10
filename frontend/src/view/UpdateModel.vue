<template>
  <div class="update-model">
    <h3>Update Model</h3>
    <div class="update-model-form">
      <div class="title">
        <span>
          {{form.modelName}}
        </span>
        <el-dropdown v-if="versionList.length>1" @command="selectVersion">
          <span class="el-dropdown-link">
            {{version}}<i class="el-icon-arrow-down el-icon--right"></i>
          </span>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item v-for="(item,index) in versionList" :key="index" :command="item">{{item}}</el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>
      </div>

      <el-form ref="form" :rules="rules" :model="form" label-width="120px" size="mini" label-position='left'>
        <el-form-item label="modelUrl:" prop="modelUrl">
          <el-input v-model="form.modelUrl" disabled></el-input>
        </el-form-item>
        <el-row :gutter="20">
          <!-- <el-col :span="8">
            <el-form-item label="modelName:" prop="modelName">
              <el-input v-model="form.modelName" disabled></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="version:">
              <el-input v-model="version" disabled></el-input>
            </el-form-item>
          </el-col> -->

          <el-col :span="8">
            <el-form-item label="status:" prop="status">
              <el-input v-model="form.status" disabled></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="loadedAtStartup:" prop="loadedAtStartup">
              <el-switch v-model="form.loadedAtStartup" disabled></el-switch>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="batch_size:" prop="batchSize">
              <el-input v-model.number="form.batchSize"></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="max_batch_delay:" prop="maxBatchDelay">
              <el-input v-model.number="form.maxBatchDelay"></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="max_idle_time:" prop="maxIdleTime">
              <el-input v-model.number="form.maxIdleTime"></el-input>
            </el-form-item>
          </el-col>

        </el-row>
      </el-form>

    </div>
    <div class="worker-list">
      <div class="title">Worker Groups</div>
      <el-tabs v-model="activeDevice">
        <el-tab-pane :label="item.device.deviceType+':'+item.device.deviceId" :name="index+''" v-for="(item,index) in form.workerGroups" :key="index">
          <el-row :gutter="20">
            <el-col :span="8">
              min_worker:
              <el-input-number v-model="form.workerGroups[activeDevice].minWorkers" size="small" :min="1" :max="100"></el-input-number>
            </el-col>
            <el-col :span="8">
              max_worker:
              <el-input-number v-model="form.workerGroups[activeDevice].maxWorkers" size="small" :min="1" :max="100"></el-input-number>
            </el-col>
          </el-row>
        </el-tab-pane>

      </el-tabs>
      <ul>
        <li :class="['worker-item', {'active':activeIndex==index}]" v-for="(item,index) in form.workerGroups[0].workers" :key="index" @click="activeIndex=index"><img src="../assets/img/worker.png" alt="" srcset=""> worker</li>

      </ul>
      <div class="worker-info">
        <el-row :gutter="20">

          <el-col :span="9">id: {{form.workerGroups[activeDevice].workers[activeIndex].id}}</el-col>
          <el-col :span="9">deviceType: {{form.workerGroups[activeDevice].device.deviceType}}</el-col>
          <el-col :span="9">deviceId: {{form.workerGroups[activeDevice].device.deviceId}}</el-col>
          <el-col :span="9">status: {{form.workerGroups[activeDevice].workers[activeIndex].status}}</el-col>
          <el-col :span="9">startTime: {{form.workerGroups[activeDevice].workers[activeIndex].startTime|dateFormat}}</el-col>
        </el-row>
      </div>
      <div class="submit-btns">
        <el-button type="info" size="medium" @click="cancel">cancel</el-button>
        <el-button type="primary" size="medium" @click="submit">submit</el-button>
      </div>
    </div>
  </div>
</template>

<script>
import * as modelApi from "@/api/modelAPI"

export default {
  name: "UpdateModel",
  components: {

  },
  props: {

  },
  data() {
    return {
      version: "",
      versionList: "",
      form: {
        modelName: "res18",
        modelUrl: "file:/D:/DeepLearning/traced_resnet18",

        batchSize: 1,
        maxBatchDelay: 100,
        maxIdleTime: 60,
        queueLength: 0,
        status: "Healthy",
        loadedAtStartup: false,
        workerGroups: [
          {
            workers: [{
              "id": 1,
              "startTime": "2022-07-29T16:10:20.445Z",
              "status": "READY"
            }],
            "device": {
              "deviceType": "cpu",
              "deviceId": -1
            },
            minWorkers: 1,
            maxWorkers: 10,
          },
          {
            workers: [{
              "id": 1,
              "startTime": "2022-07-29T16:10:20.445Z",
              "status": "READY"
            }],
            "device": {
              "deviceType": "gpu",
              "deviceId": 0
            },
            minWorkers: 1,
            maxWorkers: 16,
          }
        ],

      },
      activeIndex: 0,
      rules: {
        batchSize: [{ type: 'number', message: 'Batch size must be a number' }],
        maxBatchDelay: [{ type: 'number', message: 'Max batch delay must be a number' }],
        maxIdleTime: [{ type: 'number', message: 'Max idle time must be a number' }],
      },
      models: [],
      activeDevice: '0'
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    let modelName = this.$route.params.name
    let name = modelName.split(":")[0]
    let version = modelName.split(":")[1]
    let res = await modelApi.modelInfo(name)
    this.models = res
    let model = res.find(v => v.version == version || (!version && !v.version))
    console.log('model',model);
    this.versionList = res.map(v => v.version)
    this.form = model.models[0]
    this.version = model.version
    console.log(res);
  },
  methods: {
    async submit() {
      let flag = true
      await this.$refs.form.validate(async (valid, rules) => {
        if (!valid) {
          for (const key in rules) {
            flag = false;
            console.log(key, rules[key][0].message)
            this.$message.error(rules[key][0].message)
            throw Error(rules[key][0].message)
          }
          return
        }
      })
      if (!flag) {
        throw Error('Valid failed')
      }


    },
    selectVersion(ver) {
      this.version = ver
      let model = this.models.find(v => v.version == ver)
      this.form = model.models[0]
      this.activeIndex = 0
    },
    cancel() {
      this.$router.go(-1)
    },
  },
};
</script>

<style  lang="less">
.update-model {
  height: calc(100vh - 110px);
  display: flex;
  flex-direction: column;
  // overflow: auto;
  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin-top: 0px;
  }
  .title {
    background: #d6f2ff;
    height: 40px;
    display: flex;
    align-items: center;
    font-size: @titleSize4;
    justify-content: space-between;
    padding: 0px 40px;
    position: relative;
    &::before {
      position: absolute;
      content: "";
      width: 10px;
      height: 10px;
      background: @themeColor;
      border-radius: 10px;
      display: block;
      top: 16px;
      left: 15px;
    }
  }
  .update-model-form {
    background: #fff;
    // padding: 20px 40px;
    // padding-bottom: 20px;
    // height: calc(100vh - 160px);

    .el-form {
      padding: 20px 40px;

      .el-select {
        width: 100%;
      }
      .el-collapse {
        border: none;
        .el-collapse-item {
          .el-collapse-item__header {
            background: #e4f6ff;
            border: none;
            justify-content: center;
            color: @themeColor;
            margin-bottom: 20px;
            height: 40px;
            .el-collapse-item__arrow {
              display: none;
            }
            [class^="icon-"] {
              margin-right: 10px;
            }
          }
          .el-collapse-item__wrap {
            border: none;
          }
        }
      }
    }
    
  }
  .worker-list {
    margin-top: 20px;
    background: #fff;
    // padding: 20px 40px;
    .el-tabs {
      font-size: @textSize;
      padding: 10px 40px;
      padding-bottom: 0;
      .el-input-number{
        margin-left: 10px;
      }
    }
    ul {
      display: flex;
      flex-wrap: wrap;
      margin: 0;
      padding: 20px 40px;
      .worker-item {
        margin-right: 20px;
        width: 200px;
        height: 100px;
        background: #c9eeff;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        color: @themeColor;
        font-size: @titleSize2;
        cursor: pointer;
      }
      .worker-item.active {
        box-sizing: border-box;
        border: 1px solid rgb(69, 255, 69);
      }
    }
    .worker-info {
      color: #3c4353;
      font-size: @textSize;
      padding: 0 40px;
      .el-col {
        margin-bottom: 20px;
      }
    }
    
  }
  .submit-btns {
      text-align: right;
      margin-right: 40px;
      button {
        width: 240px;
        border-radius: 10px;
      }
    }
}
</style>
