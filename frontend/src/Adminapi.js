import { API_BASE_URL } from "./config";

// Thin wrapper around fetch for the admin-only endpoints. Throws a
// distinguishable error on 401/403 so callers can drop back to the
// token-entry screen rather than showing a raw fetch failure.
async function adminRequest(path, token, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    },
  });

  if (res.status === 401 || res.status === 403) {
    const err = new Error("Invalid admin token");
    err.isAuthError = true;
    throw err;
  }
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || "";
    } catch {
      // response wasn't JSON -- fall back to the generic message below
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export function fetchUsers(token) {
  return adminRequest("/api/admin/users", token);
}

export function fetchUserSessions(token, userId) {
  return adminRequest(`/api/admin/users/${encodeURIComponent(userId)}/sessions`, token);
}

export function fetchSessionMessages(token, sessionId) {
  return adminRequest(`/api/admin/sessions/${encodeURIComponent(sessionId)}/messages`, token);
}

export function fetchDocuments(token) {
  return adminRequest("/api/admin/documents", token);
}

export function deleteDocument(token, source) {
  return adminRequest(`/api/admin/documents/${encodeURIComponent(source)}`, token, {
    method: "DELETE",
  });
}

export function ingestFiles(token, files) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  return adminRequest("/api/admin/ingest", token, {
    method: "POST",
    body: formData,
  });
}
