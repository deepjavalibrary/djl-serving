const path = require("path");

module.exports = {
  productionSourceMap: false,
  publicPath: '/console',
  outputDir: '../src/main/resources/static/console',
  devServer: {
    port: 8082,
    proxy: {
      '/api': {
        target: 'http://localhost:8080/',
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  },
  chainWebpack: config => {
    config
      .plugin('html')
      .tap(args => {
        args[0].title = 'DJLServing'
        return args
      })
  },
  pluginOptions: {
    'style-resources-loader': {
      preProcessor: 'less',
      patterns: [path.resolve(__dirname, "src/assets/css/common.less")]
    }
  }
}
