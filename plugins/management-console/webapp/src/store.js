// register vuex
import Vuex from 'vuex'
import Vue from 'vue'
import * as logAPI from '@/api/logAPI'
import * as env from './env'


Vue.use(Vuex)

var store = new Vuex.Store({
  state: { // this.$store.state.***
    predictionUrl: ""
  },
  mutations: { // this.$store.commit('method name', 'provide if necessary')
    savePredictionUrl(state, predictionUrl) {
      state.predictionUrl = predictionUrl;
    }
  },
  getters: { // this.$store.getters.***  
    getPredictionUrl: async (state) => {
      var predictionUrl = state.predictionUrl;
      if (!predictionUrl) {
        try {
          predictionUrl = await logAPI.inferenceAddress()
          let port = getPort(predictionUrl)
          if(port == window.location.port){
            predictionUrl = env.baseUrl
          }else{
            predictionUrl = window.location.protocol+"//"+window.location.hostname+":"+port
          }
        } catch (error) {
          predictionUrl = env.baseUrl
          console.log("getPredictionUrl", error);
        }
        store.commit("savePredictionUrl", predictionUrl.replace("\n", ""))
        return predictionUrl

      } else {
        return predictionUrl;
      }
    },
  },
  actions: {}
})
function getPort(url) {
  url = url.match(/^(([a-z]+:)?(\/\/)?[^\/]+).*$/)[1] || url;
  var parts = url.split(':'),
    port = parseInt(parts[parts.length - 1], 10);
  if (parts[0] === 'http' && (isNaN(port) || parts.length < 3)) {
    return 80;
  }
  if (parts[0] === 'https' && (isNaN(port) || parts.length < 3)) {
    return 443;
  }
  if (parts.length === 1 || isNaN(port)) return 80;
  return port;
}

export default store