// 导入格式化时间的插件
import moment from 'moment'
// 定义全局的过滤器
export function dateFormat (dataStr, pattern = "YYYY-MM-DD HH:mm:ss") {
  if(dataStr ==-1 ||!dataStr) return ""
  return moment(dataStr).format(pattern)
}


