import http from '../common/http'

export const getConfig = () =>{
  return http.requestGet('/config');
}
export const getVersion = () =>{
  return http.requestGet('/version');
}
export const modifyConfig = (param) =>{
  return http.requestPost('/config',param);
}
