import http from '../common/http'

export const dependencics = () =>{
  return http.requestGet('/dependency');
}
export const addDependency = (metadata) =>{
  return http.requestQuickPost('/dependency?metadata='+metadata);
}
export const delDependency = (name) =>{
  return http.requestDelete('/dependency/'+name);
}