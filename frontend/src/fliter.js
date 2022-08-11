// time formatter plugin
import moment from 'moment'
// Global filter on time
export function dateFormat (dataStr, pattern = "YYYY-MM-DD HH:mm:ss") {
  if(dataStr ==-1 ||!dataStr) return ""
  return moment(dataStr).format(pattern)
}


