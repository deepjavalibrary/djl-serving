<template>
  <div class="config">
    <h3>Configuration</h3>
    <div class="config-box">
      <div class="title">config.properties</div>
      <div class="config-content">
        <div ref="container" class="monaco-editor" style="height:100%"></div>
      </div>
    </div>
    <div class="submit-btns">
      <el-button type="info" size="medium" @click="cancel">Cancel</el-button>
      <el-button type="primary" size="medium" @click="submit">Save</el-button>
    </div>
  </div>
</template>

<script>
import * as monaco from 'monaco-editor'

export default {
  name: "Config",
  components: {

  },
  props: {

  },
  data() {
    return {
      // Main configuration
      defaultOpts: {
        value: '', // Editor values
        theme: 'vs-dark', // Editor theme：vs, hc-black, or vs-dark
        roundedSelection: true, // The editor preview box is not displayed on the right
        autoIndent: true, // Auto indent
        language: 'ini' 
      },
      // Editor object
      monacoEditor: {},
    };
  },
  computed: {

  },
  watch: {

  },
  created() {

  },
  async mounted() {
    this.init()
  },
  methods: {
    init() {
      // Initialize the contents of the container and destroy the previously generated editor
      this.$refs.container.innerHTML = ''
      // Build editor configuration
      let editorOptions =this.defaultOpts

      // Initialize editor instance
      this.monacoEditor = monaco.editor.create(this.$refs.container, editorOptions)
      // Triggered when the editor content changes
      this.monacoEditor.onDidChangeModelContent(() => {
        // this.$emit('change', this.monacoEditor.getValue())
      })
      let text = ''
      this.monacoEditor.setValue(`${text}`)

    },

    cancel() {
      this.$router.go(-1)
    },
    submit() {
      console.log(this.monacoEditor.getValue());
    }

  },
};
</script>

<style  lang="less">
.config {
  h3 {
    font-size: @titleSize2;
    font-weight: normal;
    margin-top: 0px;
  }
  .config-box {
    background: #fff;
    .config-content {
      height: calc(100vh - 250px);
    }
  }
  .title {
    background: #e5ffee;
    height: 40px;
    display: flex;
    align-items: center;
    font-size: @titleSize4;
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
  .submit-btns {
    text-align: right;
    margin-top: 20px;
    button {
      width: 240px;
      border-radius: 10px;
    }
  }
}
</style>
