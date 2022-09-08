import http from '../common/http'

export const logs = () =>{
  return http.requestGet('/console/api/logs');
}
export const inferenceAddress = () =>{
  return http.requestGet('/console/api/inferenceAddress');
}
export const logInfo = (name) =>{
  return http.requestGet('/console/api/logs/' + name);
}
export const download = (name) =>{
  return http.requestGetFile('/console/api/logs/download/' + name);
}
