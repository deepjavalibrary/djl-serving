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
                <el-menu-item index="dependency">Dependency</el-menu-item>
                <el-menu-item index="config">Config</el-menu-item>
              </el-submenu>
            </el-menu>
          </div>
          <div class="right">0.19.0</div>
        </div>
      </el-header>
      <el-main >
        <div class="w1200">

          <router-view></router-view>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
  name: "Main",
  components: {

  },
  props: {

  },
  data() {
    return {
      activeIndex: '/home',
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  mounted() {

  },
  methods: {

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
        .el-submenu.is-active{
          .el-submenu__title{
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
  }
  .el-main {
    color: @textFontColor;
    padding: 20px 0px;
  }
}
</style>
