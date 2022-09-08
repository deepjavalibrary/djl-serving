import http from '../common/http'

export const dependencies = () =>{
  return http.requestGet('/console/api/dependency');
}
export const addDependency = (params) =>{
  return http.requestPostForm('/console/api/dependency', params);
}
export const delDependency = (name) =>{
  return http.requestDelete('/console/api/dependency/' + name);
}
