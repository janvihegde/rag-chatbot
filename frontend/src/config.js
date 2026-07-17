// Points at the FastAPI backend. Override at build/dev time by setting
// VITE_API_BASE_URL in frontend/.env (e.g. VITE_API_BASE_URL=http://localhost:8000).
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";