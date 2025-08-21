export function getSessionId(){
  let sid = localStorage.getItem('smartdoc_session_id');
  if(!sid){
    sid = String(Math.floor(Date.now()/1000));
    localStorage.setItem('smartdoc_session_id', sid);
  }
  return Number(sid);
}
