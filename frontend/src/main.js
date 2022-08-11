import Vue from 'vue'
import App from './App.vue'
import router from './router'
import '@/plugins/element.js'
//filter
import * as filters from "./fliter"
// vuex
import store from './store'


// import global css
import '@/assets/css/global.css'
// Import font
import '@/assets/css/iconfont.css'


Vue.config.productionTip = false

// global filter
Object.keys(filters).forEach(key => {
  Vue.filter(key, filters[key])
})

new Vue({
  router,
  render: h => h(App),
  store
}).$mount('#app')
