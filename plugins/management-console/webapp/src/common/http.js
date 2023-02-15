
import axios from 'axios'
import * as env from '../env'

axios.defaults.baseURL = env.baseUrl //root path
axios.defaults.withCredentials = false //cross
axios.defaults.timeout = 600000
// axios.defaults.headers.post['Content-Type'] = 'application/x-www=form-urlencoded'

import { Loading } from 'element-ui';
import { Message } from 'element-ui';
let axiosInst = axios.create({});//axios instance
var loadingInstance;
// request interceptors
axiosInst.interceptors.request.use(
  req => {

    let data = req.data || {}
    if(!data.cancelLoading){
      loadingInstance = Loading.service();
    }
    if(req.headers.updateBaseURL){
      req.baseURL = req.headers.updateBaseURL
    }else{
      req.baseURL =  env.baseUrl
    }
    delete req.headers.updateBaseURL
    console.log("req",req);
    // console.log("headers",req.headers);
    return req
  },
  err => {
    return Promise.reject(err)
  }
);
//response interceptors
axiosInst.interceptors.response.use(response => {

  loadingInstance.close();
  // console.log("response",response);

  return response;
}, error => {
  loadingInstance.close();
    console.log("error",error.response);
    if(error.response.data instanceof Blob){
      var reader = new FileReader();
      reader.readAsText(error.response.data, 'utf-8');
      reader.onload = function (e) {
        Message.error({ message: JSON.parse( reader.result).message, customClass: 'error-toast' });
          // console.info(reader.result);
      }
    }else{
      Message.error({ message: error.response.data.message, customClass: 'error-toast' });
    }

  return Promise.reject(error);
  // return error;
});


export default {
  //get
  requestGet(url, params = {},headers) {
    return new Promise((resolve, reject) => {
      axiosInst.get(url,{ params, headers}).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  },
  //get without param
  requestQuickGet(url) {
    return new Promise((resolve, reject) => {
      axiosInst.get(url).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  },
  //get file
  requestGetFile(url, params,progress) {
    return new Promise((resolve, reject) => {
      axiosInst({
        methods: 'get',
        url,
        params,
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json'
        },
        onDownloadProgress: (evt) => {
          let downProgress =parseInt((evt.loaded / evt.total) * 100)
          if(typeof progress === "function")  progress(downProgress)
        }
      }).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  },
  //post
  requestPost(url, params = {},header={},responseType="") {
    return new Promise((resolve, reject) => {
      axiosInst.post(url, params,{
		responseType: responseType,
        headers:{...header},
        // myHeader:header
      }).then(res => {
        resolve(res)
      }).catch(error => {
        // console.log("requestPost-error",error.response);
        reject(error.response.data)
      })
    })
  },
  //post
  requestPostForm(url, params = {},responseType="",header) {
    return new Promise((resolve, reject) => {
      axiosInst.post(url, params, {
        responseType: responseType,
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          ...header
        },
        // myHeader:header
      }).then(res => {
        // console.log("requestPostForm",res);
        resolve(res)
      }).catch(error => {
        console.log("requestPostForm-error",error.response);
        reject(error.response.data)
      })
    })
  },
  //post
  requestQuickPost(url, params = {}) {
    return new Promise((resolve, reject) => {
      url = url + "?"+Object.keys(params).map(v => v+"="+(params[v] == undefined ? '' : params[v])).join("&")

      axiosInst.post(url, {}, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
      }).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  },
  //put
  requestPut(url, params = {}) {
    return new Promise((resolve, reject) => {
      axiosInst.put(url, params).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  },
  //delete
  requestDelete(url, params = {}) {
    return new Promise((resolve, reject) => {
      axiosInst.delete(url, params).then(res => {
        resolve(res.data)
      }).catch(error => {
        reject(error)
      })
    })
  }
}
