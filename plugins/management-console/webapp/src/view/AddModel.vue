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
                <el-option label="MXNet" value="MXNet"></el-option>
                <el-option label="PyTorch" value="PyTorch"></el-option>
                <el-option label="TensorFlow" value="TensorFlow"></el-option>
                <el-option label="PaddlePaddle" value="PaddlePaddle"></el-option>
                <el-option label="Python" value="Python"></el-option>
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="device:" prop="device">
              <template slot="label">
                <el-tooltip class="item" effect="dark" content="the GPU device id to load the model. The default is CPU (`-1')." placement="top"><span>Device:</span></el-tooltip>
              </template>

              <el-input v-model.number="form.device"></el-input>
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
                <el-form-item label="batch_size:" prop="batch_size">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the inference batch size. The default value is `1`." placement="top"><span>Batch size:</span></el-tooltip>
                  </template>
                  <el-input v-model.number="form.batch_size"></el-input>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_batch_delay:" v-if="form.batch_size>1" prop="max_batch_delay">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the maximum delay for batch aggregation. The default value is 100 milliseconds." placement="top"><span>Max batch delay:</span></el-tooltip>
                  </template>
                  <el-input v-model.number="form.max_batch_delay"></el-input>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_idle_time:" v-if="form.batch_size>1" prop="max_idle_time">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content=" the maximum idle time before the worker thread is scaled down." placement="top"><span>Max idle time:</span></el-tooltip>
                  </template>
                  <el-input v-model.number="form.max_idle_time"></el-input>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="min_worker:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the minimum number of worker processes. The default value is `1`." placement="top"><span>Min worker:</span></el-tooltip>
                  </template>
                  <el-input-number v-model="form.min_worker" :min="1" :max="10" label="min_worker"></el-input-number>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="max_worker:">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="the maximum number of worker processes. The default is the same as the setting for `min_worker`." placement="top"><span>Max worker:</span></el-tooltip>
                  </template>
                  <el-input-number v-model="form.max_worker" :min="1" :max="10" label="max_worker"></el-input-number>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="synchronous:" prop="synchronous">
                  <template slot="label">
                    <el-tooltip class="item" effect="dark" content="whether or not the creation of worker is synchronous. The default value is true." placement="top"><span>Synchronous:</span></el-tooltip>
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
        model_name: '',
        model_version: '0.0.1',
        engine: 'MXNet',
        device: -1,
        synchronous: true,
        batch_size: 1,
        max_batch_delay: 100,
        max_idle_time: 60,
        min_worker: 1,
        max_worker: 1
      },
      rules: {
        url: [{ required: true, message: 'Url cannot be empty', trigger: 'blur' }],
        model_name: [{ required: true, message: 'Model name cannot be empty', trigger: 'blur' }],
        batch_size: [{ type: 'number', message: 'Batch size must be a number' }],
        device: [{ type: 'number', message: 'Device id must be a number' }],
        max_batch_delay: [{ type: 'number', message: 'Max batch delay must be a number' }],
        max_idle_time: [{ type: 'number', message: 'Max idle time must be a number' }],
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
      this.$message.success('Model "' + this.form.model_name + '" registered.')
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
