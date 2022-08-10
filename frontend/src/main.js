import Vue from 'vue'
import App from './App.vue'
import router from './router'
import '@/plugins/element.js'
//过滤器
import * as filters from "./fliter"
// vuex
import store from './store'

// 导入字体图标
// import './assets/fonts/iconfont.css'
// 导入全局样式表
import '@/assets/css/global.css'
import '@/assets/css/iconfont.css'


Vue.config.productionTip = false

// 过滤器
Object.keys(filters).forEach(key => {
  Vue.filter(key, filters[key])
})

new Vue({
  router,
  render: h => h(App),
  store
}).$mount('#app')
