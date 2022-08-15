// register vuex
import Vuex from 'vuex'
import Vue from 'vue'

Vue.use(Vuex)

var store = new Vuex.Store({
  state: { // this.$store.state.***
   
  },
  mutations: { // this.$store.commit('method name', 'provide if necessary')
    
  },
  getters: { // this.$store.getters.***  
   
  },
  actions: {}
})

export default store