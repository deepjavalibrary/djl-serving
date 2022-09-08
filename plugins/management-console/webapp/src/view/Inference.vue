<template>
  <div class="inference">
    <h3>Inference</h3>
    <div class="inference-model-form">
      <div class="title"><span>{{modelName}}</span>
        <el-dropdown v-if="versionList.length>1" @command="selectVersion">
          <span class="el-dropdown-link">
            {{activeVersion}}<i class="el-icon-arrow-down el-icon--right"></i>
          </span>
          <el-dropdown-menu slot="dropdown">
            <el-dropdown-item v-for="(item,index) in versionList" :key="index" :command="item">{{item}}</el-dropdown-item>
          </el-dropdown-menu>
        </el-dropdown>
      </div>
      <el-form ref="form" label-width="120px" size="mini" label-position='left'>
        <div class="header">Headers</div>

        <el-row :gutter="20" v-for="(header,index) in headers" :key='index'>
          <el-checkbox v-model="header.checked" :disabled="header.disabled">
          </el-checkbox>
          <el-col :span="6">
            <el-input size="mini" v-model="header.key" @input='headerChange'></el-input>
          </el-col>
          <span>:</span>
          <el-col :span="6">
            <el-autocomplete size="mini" class="inline-input" @input='headerChange' v-if="header.key == 'content-type'" v-model="header.value" :fetch-suggestions="querySearch" placeholder="Please select content type" clearable></el-autocomplete>
            <el-input size="mini" v-model="header.value" @input='headerChange' clearable v-else></el-input>
          </el-col>
          <i class="el-icon-close" @click="deleteHeader(index)" v-if="index != headers.length-1"></i>
        </el-row>
        <div class="header">Body</div>
        <el-form-item label="Data type:">
          <el-radio-group v-model="dataType" @change="dataTypeChange">
            <el-radio label="file">File</el-radio>
            <el-radio label="text">Text</el-radio>
          </el-radio-group>
        </el-form-item>
        <div class="upload-area" v-show='dataType=="file"'>
          <el-upload ref='upload' multiple :show-file-list="false" :on-change="onChange" :auto-upload="false" name="data" action="">
            <el-button size="medium" type="success">Click to upload</el-button>
          </el-upload>
          <div class="file-list">

            <div class="file" v-for="(file,index) in fileList " :key="index">
              <div class="file-left">
                <img v-if="['xls','csv'].includes(file.suffName)" src="../assets/img/xls.png" alt="" srcset="">
                <img v-else-if="file.suffName.indexOf('ppt')>-1" src="../assets/img/ppt.png" alt="" srcset="">
                <img v-else-if="imgs.includes(file.suffName)" src="../assets/img/jpg.png" alt="" srcset="">
                <img v-else-if="file.suffName.indexOf('pdf')>-1" src="../assets/img/pdf.png" alt="" srcset="">
                <img v-else-if="['rar','zip'].includes(file.suffName)" src="../assets/img/zip.png" alt="" srcset="">
                <img v-else src="../assets/img/doc.png" alt="" srcset="">
                <span class="file-name">{{file.fileName}}</span>
              </div>
              <i class="el-icon-close" @click="removeFile(file)"></i>
            </div>

          </div>
          <el-row :gutter="20" v-for="(data,index) in bodyDatas" :key='index'>
            <el-checkbox v-model="data.checked" :disabled="data.disabled">
            </el-checkbox>
            <el-col :span="6">
              <el-input size="mini" v-model="data.key" @input='dataChange'></el-input>
            </el-col>
            <span>:</span>
            <el-col :span="6">
              <el-input size="mini" v-model="data.value" @input='dataChange' clearable></el-input>
            </el-col>
            <i class="el-icon-close" @click="deleteHeader(index)" v-if="index != headers.length-1"></i>
          </el-row>
        </div>
        <div class="text-area" v-show='dataType=="text"'>
          <el-input type="textarea" :autosize="{ minRows: 3, maxRows: 6}" v-model="bodyText">
          </el-input>
        </div>
        <div class="submit-btns">
          <el-button type="info" size="medium" @click="cancel">Cancel</el-button>
          <el-button type="primary" class="predict" size="medium" @click="predict">Predict</el-button>
        </div>
      </el-form>
    </div>
    <div class="result-box">
      <div :class="['title',{'error':resultError}]">Result</div>
      <div class="result-content">
        <pre>{{resultText}}</pre>
        <img :src="imgSrc">
      </div>
    </div>
  </div>
</template>

<script>
import * as modelApi from "@/api/modelAPI"
export default {
  name: "Inference",
  components: {

  },
  props: {

  },
  data() {
    return {
      modelName: "",
      dataType: 'file',
      bodyText: "",
      headers: [{ key: 'Accept', value: '', checked: false, disabled: false }, { key: 'content-type', value: '', checked: false, disabled: true }, { key: '', value: '', checked: true, disabled: false }],
      contentList: [{ value: 'text/string' }, { value: 'tensor/npz' }, { value: 'tensor/ndlist' }, { value: 'application/json' }, { value: 'image/jpg' }],
      fileList: [],
      imgs: ['jpg', 'jpeg', 'png', 'bmp'],
      versionList: [],
      activeVersion: '',
      resultText: "",
      imgSrc: "",
      bodyDatas: [{ key: '', value: '', checked: true }],
      resultError: false

    };
  },
  computed: {

  },
  watch: {

  },
  async created() {
    console.log(this.$route.params.name);
    let modelName = this.$route.params.name
    this.modelName = modelName.split(":")[0]
    let version = modelName.split(":")[1]
    let res = await modelApi.modelInfo(this.modelName)
    this.versionList = res.map(v => v.version)
    this.activeVersion = version
    console.log(res);
  },
  mounted() {

  },
  methods: {
    querySearch(queryString, cb) {
      var contentList = this.contentList;
      var results = queryString ? contentList.filter(v => v.value.includes(queryString)) : contentList;
      cb(results);
    },
    headerChange(val) {
      console.log("headerChange", val);
      if (val) {
        let header = this.headers[this.headers.length - 1]
        if (header.key || header.value) {
          this.headers = [...this.headers, { key: '', value: '', checked: true, disabled: false }]
        }
      }
    },
    dataChange(val) {
      console.log("dataChange", val);
      if (val) {
        let data = this.bodyDatas[this.bodyDatas.length - 1]
        if (data.key || data.value) {
          this.bodyDatas = [...this.bodyDatas, { key: '', value: '', checked: true }]
        }
      }
    },
    deleteHeader(index) {
      this.headers.splice(index, 1)
    },
    removeFile(file) {
      return this.$confirm(`Are you sure to delete  ${file.fileName}？`, 'Warning', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel',
        type: 'warning',
      }).then(() => {

        this.fileList = this.fileList.filter(f => f.uid != file.uid)
        this.$refs.upload.uploadFiles = this.$refs.upload.uploadFiles.filter(f => f.uid != file.uid)
      })

    },
    onChange(file, fileList) {
      console.log("onChange", file);
      this.fileList = fileList.map(f => {
        let file = { ...f }
        file.fileName = f.name
        file.suffName = f.name.substring(file.name.lastIndexOf('.') + 1)
        return file
      })
    },
    async predict() {
      this.resultError = false
      this.imgSrc = ""
      this.resultText = ''
      let fileData = new FormData()
      if (this.dataType == 'file') {
        if (this.fileList.length < 1) {
          this.$message.error("File list is empty ")
          return
        }
        this.fileList.forEach(file => {
          fileData.append("data", file.raw)
        })
        this.bodyDatas.forEach(v => {
          if (v.checked && v.key) {
            fileData.append([v.key], v.value)
          }
        })
      } else {
        if (!this.bodyText) {
          this.$message.error("Text body is empty ")
          return
        }
        fileData = this.bodyText
      }
      let header = {}
      this.headers.forEach(v => {
        if (v.checked && v.key) {
          header[v.key] = v.value
        }
      })
   
      if (process.env.NODE_ENV != 'development') {
        let url = await this.$store.getters.getPredictionUrl
        let inferenceFlag = this.$store.getters.getInferenceFlag
        if (!inferenceFlag) {
          let str = "Since 'inference_address' is inconsistent with 'management_address', please confirm whether cors_allowed configuration is enabled"
          this.$message.error(str)
          return
        }
        header.updateBaseURL = url
      }
      let res
      try {
        res = await modelApi.predictions(this.modelName, this.activeVersion, fileData, header)
      } catch (error) {
        this.resultError = true
        if (error.code) {
          this.resultText = error.message
        } else if (error instanceof Blob) {
          let blob = new Blob([error])
          var reader = new FileReader()
          reader.onload = e => {
            this.resultText = JSON.parse(reader.result).message
          }
          reader.readAsText(blob)
        }
        return
      }
      if (res.headers && res.headers['content-type']) {
        let contentType = res.headers['content-type']
        console.log("contentType", contentType);
        if (contentType.startsWith("text/") || contentType == 'application/json') {
          if (res.data instanceof Blob) {
            let blob = new Blob([res.data])
            var reader = new FileReader()
            reader.onload = e => {
              this.resultText = e.target.result
            }
            reader.readAsText(blob)
          } else {
            this.resultText = res.data
          }
        } else if (['tensor/ndlist', 'tensor/npz', "multipart/form-data"].includes(contentType)) {
          const link = document.createElement('a');
          let blob = new Blob([res.data]);
          link.style.display = 'none';
          link.href = URL.createObjectURL(blob);
          link.setAttribute('download', contentType.replace('/', "."));
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link)
        } else if (contentType.startsWith("image/")) {
          let blob = new Blob([res.data])
          this.imgSrc = window.URL.createObjectURL(blob)
        }
      }
      console.log(res);
    },
    cancel() {
      this.$router.go(-1)
    },
    selectVersion(v) {
      this.activeVersion = v
    },
    dataTypeChange(val) {
      console.log("dataTypeChange", val);
      let find = this.headers.find(v => v.key == "content-type")
      if (val == "file") {
        find.disabled = true
        find.checked = false
      } else {
        find.disabled = false
        find.checked = true
      }
    }
  },
};
</script>

<style  lang="less">
.inference {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 110px);
  .el-icon-close {
    color: #aaa;
  }

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
    .el-dropdown {
      font-size: @titleSize4;
      color: #232323;
    }
  }
  .inference-model-form {
    background: #fff;
    // padding-bottom: 20px;
    .el-form {
      padding: 0px 40px 20px 40px;
      .header {
        position: relative;
        font-size: @titleSize4;
        height: 36px;
        display: flex;
        align-items: center;
        &::before {
          position: absolute;
          content: "";
          width: 3px;
          height: 14px;
          background: @themeColor;
          display: block;
          top: 50%;
          left: -10px;
          margin-top: -6px;
        }
      }

      .el-row {
        display: flex;
        align-items: center;
        color: #232323;
        margin-bottom: 10px;
        .el-checkbox {
          margin-top: -2px;
        }
        .el-autocomplete {
          width: 100%;
        }
        .el-icon-close {
          display: none;
          cursor: pointer;
        }
      }
      .el-row:hover {
        .el-icon-close {
          display: block;
        }
      }
      .upload-area {
        padding: 20px;
        border: 1px solid #dfdfe1;
        border-radius: 5px;
        // display: flex;
        // flex-direction: column;
        .file-list {
          max-height: 150px;
          overflow-y: auto;
        }
        .ext-info {
          font-size: 12px;
          color: #a6aab8;
          padding: 20px 0;
          margin: 0;
        }
        .file {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 10px;
          .file-left {
            cursor: pointer;
            display: flex;
            align-items: center;
            width: 240px;

            img {
              width: 16px;
            }
            .file-name {
              margin-left: 10px;
              font-size: 13px;
              color: #606266;
            }
          }

          > .el-icon-close {
            font-size: 18px;
            cursor: pointer;
          }
        }
        .el-row {
          margin: 0 !important;
          margin-top: 10px !important;
        }
      }
    }
    .submit-btns {
      text-align: right;
      button {
        width: 240px;
        border-radius: 10px;
        margin-top: 20px;
      }
    }
  }
  .result-box {
    flex: 1;
    margin-top: 20px;
    background: #fff;
    min-height: 200px;
    // height: calc(100vh - 490px);
    .title {
      background: #e5ffee;
      &::before {
        background: #02f21a;
      }
    }
    .title.error {
      background: #ffd5d7;
      &::before {
        background: #fd444e;
      }
    }
    .result-content {
      padding: 20px 40px;
      background: #fff;
    }
  }
}
</style>
