const path = require("path");

module.exports = {
  devServer: {
    port: 8082,
    proxy: {

      '/api': {
        // target: 'http://18.138.254.160/api/',
        // target: 'http://54.255.165.231/api/',
        target: 'http://127.0.0.1:8080/',
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