import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, ChevronRight, RefreshCw, Trash2, UploadCloud, Users as UsersIcon, FileText } from "lucide-react";
import {
  fetchUsers,
  fetchUserSessions,
  fetchSessionMessages,
  fetchDocuments,
  deleteDocument,
  ingestFiles,
} from "./adminApi";

const TOKEN_KEY = "truelift_admin_token";

function formatTimestamp(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function TokenGate({ onUnlock }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!value.trim() || checking) return;
    setChecking(true);
    setError(null);
    try {
      // Validate by actually calling an admin endpoint, rather than just
      // storing whatever was typed -- a wrong token should fail here, not
      // silently fail later on the dashboard.
      await fetchUsers(value.trim());
      localStorage.setItem(TOKEN_KEY, value.trim());
      onUnlock(value.trim());
    } catch (err) {
      setError(err.isAuthError ? "That token was rejected." : "Couldn't reach the server.");
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="admin-gate-shell">
      <form className="admin-gate-card" onSubmit={handleSubmit}>
        <h1>Admin access</h1>
        <p>Enter the admin token to view users, chats, and documents.</p>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Admin token"
          autoFocus
        />
        {error && <div className="admin-gate-error">{error}</div>}
        <button type="submit" disabled={checking || !value.trim()}>
          {checking ? "Checking…" : "Unlock"}
        </button>
      </form>
    </div>
  );
}

function Breadcrumbs({ items }) {
  return (
    <div className="admin-breadcrumbs">
      {items.map((item, i) => (
        <span key={i} className="admin-breadcrumb-item">
          {i > 0 && <ChevronRight size={13} className="admin-breadcrumb-sep" />}
          {item.onClick ? (
            <button className="text-link-btn" onClick={item.onClick}>
              {item.label}
            </button>
          ) : (
            <span className="admin-breadcrumb-current">{item.label}</span>
          )}
        </span>
      ))}
    </div>
  );
}

function TranscriptView({ token, session, onBack, onAuthError }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchSessionMessages(token, session.session_id)
      .then((data) => setMessages(data.messages ?? []))
      .catch((err) => {
        if (err.isAuthError) onAuthError();
        else setError("Couldn't load this conversation.");
      })
      .finally(() => setLoading(false));
  }, [token, session.session_id, onAuthError]);

  return (
    <div className="admin-panel-body">
      <Breadcrumbs
        items={[
          { label: "Users", onClick: onBack.toUsers },
          { label: session.userIdShort, onClick: onBack.toSessions },
          { label: `session:${session.session_id.slice(0, 8)}` },
        ]}
      />

      {loading && <div className="admin-empty">Loading transcript…</div>}
      {error && <div className="admin-error-banner">{error}</div>}

      {!loading && !error && messages.length === 0 && (
        <div className="admin-empty">No messages in this chat yet.</div>
      )}

      <div className="transcript-list">
        {messages.map((m, i) => (
          <div key={i} className={`transcript-row ${m.role}`}>
            <div className="transcript-role">{m.role}</div>
            <div className="transcript-content">{m.content}</div>
            <div className="transcript-time">{formatTimestamp(m.timestamp)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function UserSessionsView({ token, userId, onBack, onOpenSession, onAuthError }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchUserSessions(token, userId)
      .then((data) => setSessions(data.sessions ?? []))
      .catch((err) => {
        if (err.isAuthError) onAuthError();
        else setError("Couldn't load this user's chats.");
      })
      .finally(() => setLoading(false));
  }, [token, userId, onAuthError]);

  return (
    <div className="admin-panel-body">
      <Breadcrumbs items={[{ label: "Users", onClick: onBack }, { label: userId.slice(0, 12) }]} />

      {loading && <div className="admin-empty">Loading chats…</div>}
      {error && <div className="admin-error-banner">{error}</div>}

      {!loading && !error && sessions.length === 0 && (
        <div className="admin-empty">This user has no chats yet.</div>
      )}

      <div className="admin-list">
        {sessions.map((s) => (
          <button
            key={s.session_id}
            className="admin-list-row"
            onClick={() => onOpenSession(s)}
          >
            <div className="admin-list-main">
              <div className="admin-list-title">{s.preview || "New conversation"}</div>
              <div className="admin-list-meta">
                <span className="session-tag">session:{s.session_id.slice(0, 8)}</span>
                <span className="divider">·</span>
                <span>last active {formatTimestamp(s.last_active_at)}</span>
              </div>
            </div>
            <ChevronRight size={16} className="admin-list-chevron" />
          </button>
        ))}
      </div>
    </div>
  );
}

function UsersView({ token, onOpenUser, onAuthError }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = () => {
    setLoading(true);
    setError(null);
    fetchUsers(token)
      .then((data) => setUsers(data.users ?? []))
      .catch((err) => {
        if (err.isAuthError) onAuthError();
        else setError("Couldn't load users.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(load, [token]);

  return (
    <div className="admin-panel-body">
      <div className="admin-section-header">
        <h2>Users</h2>
        <button className="icon-btn" onClick={load} aria-label="Refresh" disabled={loading}>
          <RefreshCw size={15} className={loading ? "spin" : ""} />
        </button>
      </div>

      {error && <div className="admin-error-banner">{error}</div>}
      {!loading && !error && users.length === 0 && (
        <div className="admin-empty">No users have started a chat yet.</div>
      )}

      <div className="admin-list">
        {users.map((u) => (
          <button key={u.user_id} className="admin-list-row" onClick={() => onOpenUser(u.user_id)}>
            <div className="admin-list-main">
              <div className="admin-list-title mono">{u.user_id}</div>
              <div className="admin-list-meta">
                <span>
                  {u.session_count} chat{u.session_count === 1 ? "" : "s"}
                </span>
                <span className="divider">·</span>
                <span>last active {formatTimestamp(u.last_active_at)}</span>
              </div>
            </div>
            <ChevronRight size={16} className="admin-list-chevron" />
          </button>
        ))}
      </div>
    </div>
  );
}

function DocumentsView({ token, onAuthError }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [deletingSource, setDeletingSource] = useState(null);
  const fileInputRef = useRef(null);

  const load = () => {
    setLoading(true);
    setError(null);
    fetchDocuments(token)
      .then((data) => setDocuments(data.documents ?? []))
      .catch((err) => {
        if (err.isAuthError) onAuthError();
        else setError("Couldn't load documents.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(load, [token]);

  const handleFilesChosen = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      await ingestFiles(token, files);
      load();
    } catch (err) {
      if (err.isAuthError) onAuthError();
      else setUploadError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (source) => {
    if (!window.confirm(`Remove "${source}" from the knowledge base? This can't be undone.`)) return;
    setDeletingSource(source);
    try {
      await deleteDocument(token, source);
      setDocuments((prev) => prev.filter((d) => d.source !== source));
    } catch (err) {
      if (err.isAuthError) onAuthError();
      else setError(`Couldn't delete "${source}".`);
    } finally {
      setDeletingSource(null);
    }
  };

  return (
    <div className="admin-panel-body">
      <div className="admin-section-header">
        <h2>Documents</h2>
        <button className="icon-btn" onClick={load} aria-label="Refresh" disabled={loading}>
          <RefreshCw size={15} className={loading ? "spin" : ""} />
        </button>
      </div>

      <label className={`upload-dropzone${uploading ? " uploading" : ""}`}>
        <UploadCloud size={20} />
        <div>
          <div className="upload-dropzone-title">
            {uploading ? "Uploading…" : "Upload documents"}
          </div>
          <div className="upload-dropzone-hint">PDF, HTML, or DOCX -- click to choose files</div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.html,.htm,.docx"
          multiple
          hidden
          disabled={uploading}
          onChange={handleFilesChosen}
        />
      </label>
      {uploadError && <div className="admin-error-banner">{uploadError}</div>}

      {error && <div className="admin-error-banner">{error}</div>}
      {!loading && !error && documents.length === 0 && (
        <div className="admin-empty">No documents ingested yet.</div>
      )}

      <div className="admin-list">
        {documents.map((d) => (
          <div key={d.source} className="admin-list-row document-row">
            <div className="admin-list-main">
              <div className="admin-list-title">{d.source}</div>
              <div className="admin-list-meta">
                <span>{d.chunk_count} chunks</span>
                <span className="divider">·</span>
                <span>ingested {formatTimestamp(d.ingested_at)}</span>
              </div>
            </div>
            <button
              className="icon-btn danger"
              onClick={() => handleDelete(d.source)}
              disabled={deletingSource === d.source}
              aria-label={`Delete ${d.source}`}
            >
              <Trash2 size={15} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AdminPanel({ onBackToChat }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [tab, setTab] = useState("users"); // users | documents
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [selectedSession, setSelectedSession] = useState(null);

  const handleAuthError = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
  }, []);

  if (!token) {
    return <TokenGate onUnlock={setToken} />;
  }

  const switchTab = (next) => {
    setTab(next);
    setSelectedUserId(null);
    setSelectedSession(null);
  };

  let usersContent;
  if (selectedSession) {
    usersContent = (
      <TranscriptView
        token={token}
        session={{ ...selectedSession, userIdShort: selectedUserId.slice(0, 12) }}
        onAuthError={handleAuthError}
        onBack={{
          toUsers: () => {
            setSelectedUserId(null);
            setSelectedSession(null);
          },
          toSessions: () => setSelectedSession(null),
        }}
      />
    );
  } else if (selectedUserId) {
    usersContent = (
      <UserSessionsView
        token={token}
        userId={selectedUserId}
        onBack={() => setSelectedUserId(null)}
        onOpenSession={setSelectedSession}
        onAuthError={handleAuthError}
      />
    );
  } else {
    usersContent = <UsersView token={token} onOpenUser={setSelectedUserId} onAuthError={handleAuthError} />;
  }

  return (
    <div className="app-shell">
      <div className="admin-column">
        <header className="chat-header">
          <div className="brand-mark">
            <button className="icon-btn" onClick={onBackToChat} aria-label="Back to chat">
              <ArrowLeft size={16} />
            </button>
            <div>
              <div className="brand-name">Admin dashboard</div>
              <div className="brand-subtitle">users, chats &amp; documents</div>
            </div>
          </div>
          <div className="admin-tabs">
            <button
              className={`admin-tab${tab === "users" ? " active" : ""}`}
              onClick={() => switchTab("users")}
            >
              <UsersIcon size={14} /> Users
            </button>
            <button
              className={`admin-tab${tab === "documents" ? " active" : ""}`}
              onClick={() => switchTab("documents")}
            >
              <FileText size={14} /> Documents
            </button>
          </div>
        </header>

        {tab === "users" ? usersContent : <DocumentsView token={token} onAuthError={handleAuthError} />}
      </div>
    </div>
  );
}
