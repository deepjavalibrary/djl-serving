<template>
  <div class="add-model">
    <h3>Add Model</h3>
    <div class="add-model-form">
      <div class="title">Model Info</div>

      <el-form ref="form" :rules="rules" :model="form" label-width="120px" size="mini" label-position='left'>
        <el-form-item label="url:" prop="url">
          <template slot="label">
            <el-tooltip class="item" effect="dark" content="Model url." placement="top"><span>Url:</span></el-tooltip>
          </template>
          <el-input v-model="form.url" @blur="nameBlur">
            <el-upload slot="append" :show-file-list="false" :action="baseURL+'console/api/upload'" :on-error="uploadFailed" :on-success="uploadSuccess" :before-upload="beforeUpload">
              <el-button slot="trigger" icon="el-icon-upload2"></el-button>
            </el-upload>
          </el-input>
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="model_name:" prop="model_name">
              <template slot="label">
                <el-tooltip class="item" effect="dark" content="the name of the model and workflow; this name will be used as {workflow_name} in other API as path. If this parameter is not present, modelName will be inferred by url." placement="top"><span>Model name:</span></el-tooltip>
              </template>
              <el-input v-model="form.model_name" @blur="nameBlur"></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="model_version:" prop="model_version">
              <template slot="label">
                <el-tooltip class="item" effect="dark" content="the version of the mode" placement="top"><span>Model version:</span></el-tooltip>
              </template>
              <el-input v-model="form.model_version"></el-input>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="engine:" prop="engine">
              <template slot="label">
                <el-tooltip class="item" effect="dark" content="the name of engine to load the model. The default is MXNet if the model doesn't define its engine." placement="top"><span>Engine:</span></el-tooltip>
              </template>
              <el-select v-model="form.engine">
                <el-option label="Auto detect" value=""></el-option>
                <el-option label="PyTorch" value="PyTorch"></el-option>
                <el-option label="MXNet" value="MXNet"></el-option>
                <el-option label="TensorFlow" value="TensorFlow"></el-option>
                <el-option label="PaddlePaddle" value="PaddlePaddle"></el-option>
                <el-option label="ONNXRuntime" value="OnnxRuntime"></el-option>
                <el-option label="Python" value="Python"></el-option>
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-collapse accordion @change="settingClick()">
          <el-collapse-item>
            <template slot="title">
              <i :class="{'icon-down':isDown,'icon-up':!isDown}"></i>
              Advanced Setting
            </template>
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="device:" prop="device">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the device to load the model, e.g. cpu/gpu0." placement="top"><span>Device:</span></el-tooltip>
                  </template>
                  <el-autocomplete size="mini" class="inline-input" v-model="form.device" :fetch-suggestions="listDevice" clearable></el-autocomplete>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="job_queue_size:" prop="job_queue_size">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the request job queue size, default is 1000." placement="top"><span>Job queue size:</span></el-tooltip>
                  </template>

                  <el-input-number size="mini" :min="1" v-model="form.job_queue_size"></el-input-number>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="min_worker:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the minimum number of worker processes." placement="top"><span>Min worker:</span></el-tooltip>
                  </template>
                  <el-input-number size="mini" :min="1" v-model="form.min_worker" label="min_worker"></el-input-number>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_worker:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the maximum number of worker processes." placement="top"><span>Max worker:</span></el-tooltip>
                  </template>
                  <el-input-number size="mini" :min="1" v-model="form.max_worker" label="max_worker"></el-input-number>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_idle_time:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the maximum idle time before the worker thread is scaled down." placement="top"><span>Max idle time:</span></el-tooltip>
                  </template>
                  <el-input-number size="mini" :min="1" v-model.number="form.max_idle_time"></el-input-number>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="batch_size:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the inference batch size, default is 1." placement="top"><span>Batch size:</span></el-tooltip>
                  </template>
                  <el-input-number size="mini" :min="1" v-model.number="form.batch_size"></el-input-number>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_batch_delay:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the maximum delay for batch aggregation, default is 100 milliseconds." placement="top"><span>Max batch delay:</span></el-tooltip>
                  </template>
                  <el-input-number size="mini" :min="1" v-model.number="form.max_batch_delay"></el-input-number>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row>
              <el-col :span="8">
                <el-form-item label="synchronous:" prop="synchronous">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="whether or not the creation of worker is synchronous." placement="top"><span>Synchronous:</span></el-tooltip>
                  </template>
                  <el-switch v-model="form.synchronous"></el-switch>
                </el-form-item>
              </el-col>
            </el-row>
          </el-collapse-item>

        </el-collapse>
      </el-form>
      <div class="submit-btns">
        <el-button type="info" size="medium" @click="cancel">Cancel</el-button>
        <el-button type="primary" size="medium" @click="submit">Submit</el-button>
      </div>
    </div>
  </div>
</template>

<script>
import * as modelApi from "@/api/modelAPI"
import * as env from '../env'

export default {
  name: "AddModel",
  components: {

  },
  props: {

  },
  data() {
    return {
      form: {
        url: "",
        model_name: "",
        model_version: "",
        engine: "",
        device: "",
        job_queue_size: undefined,
        synchronous: true,
        batch_size: undefined,
        max_batch_delay: undefined,
        min_worker: undefined,
        max_worker: undefined,
        max_idle_time: undefined
      },
      rules: {
        url: [{ required: true, message: 'Url cannot be empty', trigger: 'blur' }],
      },
      isDown: true,
      baseURL: "",
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  mounted() {
    this.baseURL = window.location.origin + env.baseUrl + (env.baseUrl.endsWith("/") ? "" : "/")
  },
  methods: {
    listDevice(queryString, cb) {
      var devices = [{value: "cpu"}, {value: "gpu0"}, {value: "gpu1"}, {value: "gpu2"}, {value: "gpu3"}]
      var results = queryString ? devices.filter(v => v.value.includes(queryString)) : devices;
      cb(results);
    },
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
      let param = { ...this.form }
      if (this.isDown) param = (({ url, model_name, model_version, engine, device }) => ({ url, model_name, model_version, engine, device }))(this.form)
      let res = await modelApi.addModel(param)
      console.log("submit", res);
      this.$message.success(res.status)
      this.$router.go(-1)
    },
    settingClick() {
      this.isDown = !this.isDown

    },
    cancel() {
      this.$router.go(-1)
    },
    uploadSuccess(response, file, fileList) {
      console.log(response, file, fileList);
      this.form.url = response.replace("\n", "")
      this.loading.close()
    },
    beforeUpload(file) {
      let arr = file.name.split('.')
      console.log("beforeUpload", arr);
      let suffix = arr[arr.length - 1]
      if (!['zip', 'tar', 'gz'].includes(suffix)) {
        this.$message.error("Only files of type zip , tar or gz are allowed to be uploaded")
        return false
      }
      this.loading = this.$loading()
    },
    uploadFailed(err, file, fileList) {

      console.log("uploadFailed", err);
      if (err instanceof Error) {
        this.$message.error(err)
      } else if (err instanceof ProgressEvent) {
        this.$message.error("Make sure the 'max_request_size' value is greater than the current file size")
      }
      this.loading.close()
    },
    nameBlur() {
      this.form.model_name = this.form.model_name.replace(/\s/g, "")
      this.form.url = this.form.url.replace(/\s/g, "")
    }
  },
};
</script>

<style  lang="less">
.add-model {
  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin-top: 0px;
  }
  .add-model-form {
    background: #fff;
    height: calc(100vh - 160px);
    .title {
      background: #d6f2ff;
      font-size: @titleSize4;
      height: 40px;
      display: flex;
      align-items: center;
      padding-left: 40px;
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
    .el-form {
      padding: 20px 40px;
      .el-select {
        width: 100%;
      }
      .el-input-group__append {
        background: @themeColor;
        color: #fff;
        border: 0;
        font-size: 14px;
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
    .submit-btns {
      text-align: right;
      margin: 20px 40px;
      button {
        width: 240px;
        border-radius: 10px;
      }
    }
  }
}
</style>
