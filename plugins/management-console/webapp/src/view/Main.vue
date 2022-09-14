<template>
  <div class="main">
    <el-container>
      <el-header>
        <div class="w1200">

          <div class="left-title">
            DJL Serving
          </div>
          <div class="memu-list">
            <el-menu :default-active="activeIndex" background-color="#02a6f2" router active-text-color="#fff" text-color="#fff" class="el-menu-list" mode="horizontal">
              <el-menu-item index="/model-list">Model</el-menu-item>
              <el-menu-item index="/log-list">Log</el-menu-item>
              <!-- <el-menu-item index="/log-list">System</el-menu-item> -->
              <el-submenu index="">
                <template slot="title">System</template>
                <el-menu-item index="/dependency">Dependency</el-menu-item>
                <el-menu-item index="/config">Config</el-menu-item>
              </el-submenu>
            </el-menu>
          </div>
          <!-- <div class="right"> {{version}}</div> -->
          <el-dropdown class="right"  @command="handleCommand">
            <span class="el-dropdown-link">
              {{version}}<i class="el-icon-arrow-down el-icon--right"></i>
            </span>
            <el-dropdown-menu slot="dropdown">
              <el-dropdown-item  command="restart">Restart</el-dropdown-item>
             
            </el-dropdown-menu>
          </el-dropdown>
        </div>
      </el-header>
      <el-main>
        <div class="w1200">

          <router-view></router-view>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import * as configApi from "@/api/configAPI"

export default {
  name: "Main",
  components: {

  },
  props: {

  },
  data() {
    return {
      activeIndex: '/model-list',
      version: ''
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    let res = await configApi.getVersion()
    this.version = res.status
  },
  methods: {
    async handleCommand(command){
      if(command == 'restart'){
        let res = await configApi.restart()
        this.$message.success(res.status)
      }

    }
  },
};
</script>

<style  lang="less">
.main {
  .el-header {
    background: @themeColor;
    color: #fff;
    .w1200 {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    // padding: 0px 100px;
    .left-title {
      font-size: @titleSize1;
      font-family: Arial;
      font-weight: 400;
      // margin-right: 100px;
    }
    .memu-list {
      .el-menu-list {
        margin-bottom: 0;
        box-sizing: border-box;
        border-bottom: 0;
        .el-menu-item {
          font-size: @titleSize3;
          margin-right: 100px;
          opacity: 0.7;
          box-sizing: border-box;
          height: 60px;
        }
        .el-submenu__title {
          font-size: @titleSize3;
          opacity: 0.7;
          i {
            color: #fff;
          }
        }
        .el-submenu.is-active {
          .el-submenu__title {
            border-bottom: 0 !important;
          }
        }
        .el-menu-item.is-active {
          // font-weight: bold;
          opacity: 1;
          border-bottom: 0;
          background-color: rgba(2, 166, 242, 0) !important;
        }
        .el-menu-item:hover {
          background-color: rgba(2, 166, 242, 0) !important;
          opacity: 1;
        }
      }
    }
    .right{
      cursor: pointer;
      color: #fff;
      font-size: 16px;
      i{
        font-size: 14px;
      }
    }
  }
  .el-main {
    color: @textFontColor;
    padding: 20px 0px;
  }
}
</style>
