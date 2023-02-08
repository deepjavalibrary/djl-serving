<template>
  <div class="dependency">
    <h3>Dependency</h3>
    <div class="dependency-box">
      <div class="title">Dependency List</div>
      <div class="dependency-content">

        <div class="add-btn" @click="addFlag = true">
          <i class="el-icon-circle-plus"></i>
          <span>Add Dependency</span>
        </div>
        <el-table :data="dependencies" stripe style="width: 100%" empty-text="No data">
          <el-table-column prop="name" label="Name">
          </el-table-column>
          <el-table-column prop="groupId" label="Group Id">
            <template slot-scope="scope">
              {{scope.row.groupId||'unknown'}}
            </template>
          </el-table-column>
          <el-table-column prop="artifactId" label="Artifact Id">
            <template slot-scope="scope">
              {{scope.row.artifactId||'unknown'}}
            </template>
          </el-table-column>
          <el-table-column prop="version" label="Version">
            <template slot-scope="scope">
              {{scope.row.version||'unknown'}}
            </template>
          </el-table-column>
          <el-table-column label="Operation">
            <template slot-scope="scope">
              <el-button @click="del(scope.row)" type="text" size="small">Delete</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
    <el-dialog title="Add Dependency" class="dependency-dialog" modal-append-to-body="false" :close-on-click-modal="false" :visible.sync="addFlag" width="40%">
      <el-form ref="form" :model="form" label-width="120px" size="mini" label-position='right'>
        <el-form-item label="Type:" prop="type">
          <el-radio-group v-model="form.type">
            <el-radio :label="'engine'">Engine</el-radio>
            <el-radio :label="'jar'">Jar</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="Add form:" prop="form" v-if="form.type=='jar'">
          <el-radio-group v-model="form.from">
            <el-radio :label="'maven'">Maven</el-radio>
            <el-radio :label="'file'">File</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="Engine:" prop="engine" v-if="form.type=='engine'">
          <el-select v-model="form.engine" placeholder="">
            <el-option label="OnnxRuntime" value="OnnxRuntime"></el-option>
            <el-option label="PaddlePaddle" value="PaddlePaddle"></el-option>
            <el-option label="TFLite" value="TFLite"></el-option>
            <el-option label="XGBoost" value="XGBoost"></el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="Group Id:" prop="groupId" v-if="form.type=='jar'&&form.from=='maven'">
          <el-input v-model="form.groupId"></el-input>
        </el-form-item>
        <el-form-item label="Artifact Id:" prop="artifactId" v-if="form.type=='jar'&&form.from=='maven'">
          <el-input v-model="form.artifactId"></el-input>
        </el-form-item>
        <el-form-item label="Version:" prop="version" v-if="form.type=='jar'&&form.from=='maven'">
          <el-input v-model="form.version"></el-input>
        </el-form-item>
        <el-form-item v-if="form.type=='jar'&&form.from=='file'">

          <div class="upload-area">
            <el-upload ref='upload' multiple :show-file-list="false" :on-change="onChange" :auto-upload="false" action="">
              <el-button size="medium" type="success">Click to upload</el-button>
            </el-upload>
            <div class="file-list">

              <div class="file" v-for="(file,index) in fileList " :key="index">
                <div class="file-left">
                  <img src="../assets/img/jar.png" alt="" srcset="">
                  <span class="file-name">{{file.fileName}}</span>
                </div>
                <i class="el-icon-close" @click="removeFile(file)"></i>
              </div>
            </div>
          </div>

        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="addFlag = false">Cancel</el-button>
        <el-button type="primary" @click="addDependency">Sure</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>
import * as dependencyApi from "@/api/dependencyAPI.js"

export default {
  name: 'Dependency',
  components: {

  },
  props: {

  },
  data() {
    return {
      dependencies: [],
      addFlag: false,
      form: {
        type: 'jar',
        from: 'maven'
      },
      fileList: []
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    await this.qryDependencies()
  },
  methods: {
    async qryDependencies() {
      let res = await dependencyApi.dependencies()
      this.dependencies = res
    },
    async addDependency() {
      let params = new FormData()
      params.append("type", this.form.type)
      if (this.form.type == 'engine') {
        if (!this.form.engine) {
          this.$message.error("Please select an engine")
          return
        }
        params.append("engine", this.form.engine)
      } else {
        params.append("from", this.form.from)
        if (this.form.from == 'maven') {
          if (!this.form.groupId) {
            this.$message.error("Group Id name cannot be empty")
            return
          }
          if (!this.form.artifactId) {
            this.$message.error("Artifact Id name cannot be empty")
            return
          }
          if (!this.form.version) {
            this.$message.error("Version  name cannot be empty")
            return
          }
          params.append("groupId", this.form.groupId)
          params.append("artifactId", this.form.artifactId)
          params.append("version", this.form.version)
        } else {
          if (!this.fileList.length) {
            this.$message.error("File list is empty ")
            return
          }
          this.fileList.forEach(file => {
            params.append("data", file.raw)
          })
        }
      }
      let res = await dependencyApi.addDependency(params)
      console.log("addDependency", res);
      this.$message.success(res.data.status)
      this.addFlag = false
      await this.qryDependencies()

    },
    async del(val) {
      console.log("delete", val);
      this.$confirm(`Are you sure to delete  ${val.name}？`, 'Warning', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel',
        type: 'warning',
      }).then(async () => {
        let res = await dependencyApi.delDependency(val.name)

        this.$message.success(res.status)
        await this.qryDependencies()
      })




    },
    onChange(file, fileList) {
      console.log("onChange", file);
      let arr = file.name.split('.')
      console.log("beforeUpload", arr);
      let suffix = arr[arr.length - 1]
      if (!['jar'].includes(suffix)) {
        this.$message.error("Only files of type jar are allowed to be uploaded")
        return false
      }
      this.fileList = fileList.filter(v => v.name.endsWith('.jar')).map(f => {
        let file = { ...f }
        file.fileName = f.name
        file.suffName = f.name.substring(file.name.lastIndexOf('.') + 1)
        return file
      })
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
  },
};
</script>

<style  lang="less">
.dependency {
  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin-top: 0px;
  }
  .dependency-box {
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
  .dependency-content {
    text-align: right;
    padding: 20px;
    .add-btn {
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
  }
}
.dependency-dialog {
  .el-dialog__header {
    background: #e5ffee;
    padding: 12px 20px;
    .el-dialog__headerbtn {
      top: 16px;
    }
  }
  .upload-area {
    padding: 0 20px 20px 20px;

    // border: 1px solid #dfdfe1;
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
</style>
