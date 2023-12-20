// register vuex
import Vuex from 'vuex'
import Vue from 'vue'
import * as logAPI from '@/api/logAPI'
import * as env from './env'


Vue.use(Vuex)

var store = new Vuex.Store({
  state: { // this.$store.state.***
    predictionUrl: "",
    inferenceFlag: true,
  },
  mutations: { // this.$store.commit('method name', 'provide if necessary')
    savePredictionUrl(state, predictionUrl) {
      state.predictionUrl = predictionUrl;
    },
    saveInferenceFlag(state, inferenceFlag) {
      state.inferenceFlag = inferenceFlag;
    },
  },
  getters: { // this.$store.getters.***
    getPredictionUrl: async (state) => {
      var predictionUrl = state.predictionUrl;
      if (!predictionUrl) {
        try {
          let res = await logAPI.inferenceAddress()
          let corsAllowed = res.corsAllowed
          let port = getPort(res.inferenceAddress)
          let mgmtPort = getPort(res.inferenceAddress)
          if(port == window.location.port || port == mgmtPort){
            predictionUrl = env.baseUrl
          }else{
            if(corsAllowed !="1"){
              store.commit("saveInferenceFlag", false)
            }
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
    getInferenceFlag:  (state) => {
      return state.inferenceFlag
    }
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
