import http from '../common/http'

export const dependencics = () =>{
  return http.requestGet('/dependency');
}
export const addDependency = (params) =>{
  return http.requestPostForm('/dependency',params);
}
export const delDependency = (name) =>{
  return http.requestDelete('/dependency/'+name);
}