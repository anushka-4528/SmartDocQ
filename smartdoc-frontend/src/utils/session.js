export function getSessionId() {
  let sid = localStorage.getItem("smartdoc_session_id");
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem("smartdoc_session_id", sid);
  }
  return sid;
}
