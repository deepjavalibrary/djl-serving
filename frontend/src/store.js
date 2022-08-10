// 注册 vuex
import Vuex from 'vuex'
import Vue from 'vue'

Vue.use(Vuex)

var store = new Vuex.Store({
  state: { // this.$store.state.***
   
  },
  mutations: { // this.$store.commit('方法的名称', '按需传递唯一的参数')
    
  },
  getters: { // this.$store.getters.***  
   
  },
  actions: {}
})

export default store