// time formatter plugin
import moment from 'moment'
// Global filter on time
export function dateFormat (dataStr, pattern = "YYYY-MM-DD HH:mm:ss") {
  if(dataStr ==-1 ||!dataStr) return ""
  return moment(dataStr).format(pattern)
}

export function byteConvert (bytes) {
  if (isNaN(bytes)) {
      return '';
  }
  let symbols = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  let exp = Math.floor(Math.log(bytes)/Math.log(2));
  if (exp < 1) {
      exp = 0;
  }
  let i = Math.floor(exp / 10);
  bytes = bytes / Math.pow(2, 10 * i);

  if (bytes.toString().length > bytes.toFixed(2).toString().length) {
      bytes = bytes.toFixed(2);
  }
  return bytes + ' ' + symbols[i];
}

