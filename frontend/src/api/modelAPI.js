import http from '../common/http'


export const models = () =>{
  return http.requestGet('/models');
}
export const modelInfo = (name,version="") =>{
  return http.requestGet('/models/'+name+"/"+version);
}
export const addModel = (param) =>{
  return http.requestQuickPost('/models/',param);
}
export const delModel = (name,version="") =>{
  return http.requestDelete('/models/'+name+"/"+version);
}
export const modifyModel = (name,version="",param) =>{
  let query= "?"+Object.keys(param).map(v => v+"="+param[v]).join("&")
  return http.requestPut('/workflows/'+name+"/"+version+query);
}
export const predictions = (name,version="",param,header) =>{
  if(param instanceof FormData){
    return http.requestPostForm('/predictions/'+name+"/"+version,param,"blob");
  }else{
    return http.requestPost('/predictions/'+name+"/"+version,param,header);
  }
}
