import axios from 'axios';

// Your backend does NOT use /api/v1 prefix.
// Point to the host+port only.
export const API_BASE = 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE,
});

export default api;
