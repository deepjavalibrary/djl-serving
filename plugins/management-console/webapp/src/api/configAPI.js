import http from '../common/http'

export const getConfig = () =>{
  return http.requestGet('/console/api/config');
}
export const getVersion = () =>{
  return http.requestGet('/console/api/version');
}
export const modifyConfig = (param) =>{
  return http.requestPost('/console/api/config', param);
}
export const restart = () =>{
  return http.requestGet('/console/api/restart');
}
