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
              <el-button @click="delete(scope.row)" type="text" size="small">Delete</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
    <el-dialog title="Add Dependency" :visible.sync="addFlag" width="30%">
      <el-form ref="form" :model="form" label-width="120px" size="mini" label-position='right'>
        <el-form-item label="Type:" prop="type">
          <el-radio-group v-model="form.type">
            <el-radio :label="'engine'">Engine</el-radio>
            <el-radio :label="'jar'">Jar</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="Engine:" prop="engine" v-if="form.type=='engine'">
          <el-select v-model="form.engine" placeholder="">
            <el-option label="OnnxRuntime" value="OnnxRuntime"></el-option>
            <el-option label="PaddlePaddle" value="PaddlePaddle"></el-option>
            <el-option label="TFLite" value="TFLite"></el-option>
            <el-option label="XGBoost" value="XGBoost"></el-option>
            <el-option label="DLR" value="DLR"></el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="Group Id:" prop="groupId" v-if="form.type=='jar'">
          <el-input v-model="form.groupId"></el-input>
        </el-form-item>
        <el-form-item label="Artifact Id:" prop="artifactId" v-if="form.type=='jar'">
          <el-input v-model="form.artifactId"></el-input>
        </el-form-item>
        <el-form-item label="Version:" prop="version" v-if="form.type=='jar'">
          <el-input v-model="form.version"></el-input>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button @click="addFlag = false">cancel</el-button>
        <el-button type="primary" @click="addDependency">sure</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>
import * as dependencyApi from "@/api/dependencyAPI.js"

export default {
  components: {

  },
  props: {

  },
  data() {
    return {
      dependencies: [],
      addFlag: false,
      form: {
        type: 'jar'
      }
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
      let res = await dependencyApi.dependencics()
      this.dependencies = res
    },
    addDependency() {
      this.addFlag = true
    },
    delete(val) {

    }
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
</style>
