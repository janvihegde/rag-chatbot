import { API_BASE_URL } from "./config";

// Thin wrapper around fetch for the two admin-only endpoints. Throws a
// distinguishable error on 401/403 so the caller can drop back to the
// token-entry screen rather than showing a raw fetch failure.
async function adminFetch(path, token) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (res.status === 401 || res.status === 403) {
    const err = new Error("Invalid admin token");
    err.isAuthError = true;
    throw err;
  }
  if (!res.ok) {
    throw new Error(`Request failed (${res.status})`);
  }
  return res.json();
}

export function fetchAnalytics(token) {
  return adminFetch("/api/admin/analytics", token);
}

export function fetchEscalations(token, status = "pending") {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return adminFetch(`/api/admin/escalations${query}`, token);
}