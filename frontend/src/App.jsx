import { useCallback, useEffect, useRef, useState } from "react";
import { MessageSquare, Plus, Send, Settings } from "lucide-react";
import { API_BASE_URL } from "./config";
import { useChatStream, PIPELINE_STEPS } from "./useChatStream";
import AdminPanel from "./Adminpanel";

const USER_ID_KEY = "truelift_user_id";

// Persistent anonymous identity for this browser -- separate from
// session_id, which identifies a single chat. One user_id can own many
// sessions (see backend/app/main.py's NOTE on identity).
function getOrCreateUserId() {
  let id = localStorage.getItem(USER_ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

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

// The backend appends a trailing "(Source: ..., ...)" line to grounded
// answers (see backend/app/generation.py). We keep that data available in
// payload.citations for potential future use, but don't want it rendered
// inline in the chat bubble -- strip it here rather than changing what the
// backend sends, so citations remain available to any other UI that wants
// them later.
function stripSourceLine(text) {
  if (!text) return text;
  return text.replace(/\n*\(Source:[^)]*\)\s*$/i, "").trim();
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  const isEscalated = !isUser && msg.escalated;
  const isRejected = !isUser && (msg.scopeLabel === "off_topic" || msg.scopeLabel === "injection");
  const displayContent = isUser ? msg.content : stripSourceLine(msg.content);

  return (
    <div className={`msg-row ${isUser ? "user" : "assistant"}`}>
      <div>
        <div
          className={[
            "bubble",
            isUser ? "user" : "assistant",
            isEscalated ? "escalated" : "",
            isRejected ? "rejected" : "",
          ].join(" ").trim()}
        >
          {displayContent}
        </div>
        {isEscalated && (
          <div className="escalate-tag">
            <span>↗</span> Escalated to the team
          </div>
        )}
      </div>
    </div>
  );
}

function PipelineStatus({ node }) {
  if (!node) return null;
  return (
    <div className="pipeline-status">
      <span className="spinner" aria-hidden="true" />
      <span>{PIPELINE_STEPS[node] ?? node}</span>
    </div>
  );
}

function SessionPicker({ sessions, creating, onSelect, onNew }) {
  return (
    <div className="app-shell">
      <div className="chat-column">
        <header className="chat-header">
          <div className="brand-mark">
            <span className="status-dot" />
            <div>
              <div className="brand-name">Truelift Assistant</div>
              <div className="brand-subtitle">product &amp; support</div>
            </div>
          </div>
        </header>

        <div className="picker-body">
          <h2>Welcome back</h2>
          <p>Continue a previous conversation, or start a new one.</p>

          <button className="new-chat-btn" onClick={onNew} disabled={creating}>
            <Plus size={16} />
            {creating ? "Starting…" : "Start a new chat"}
          </button>

          <div className="session-picker-list">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                className="session-picker-item"
                onClick={() => onSelect(s.session_id)}
              >
                <MessageSquare size={15} className="session-picker-icon" />
                <div className="session-picker-text">
                  <div className="session-picker-preview">
                    {s.preview || "New conversation"}
                  </div>
                  <div className="session-picker-meta">{formatTimestamp(s.last_active_at)}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatView({ userId, sessionId, onSwitchChat, onOpenAdmin }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const { send, activeNode } = useChatStream();
  const listEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Load prior history whenever the active session changes (first mount,
  // switching to a previous chat, or starting a new one) so a refresh or
  // a chat switch doesn't lose the conversation -- the backend already
  // persists it.
  useEffect(() => {
    setMessages([]);
    fetch(`${API_BASE_URL}/api/chat/${sessionId}/history`)
      .then((r) => r.json())
      .then((data) => {
        const history = (data.history ?? []).map((m) => ({
          role: m.role,
          content: m.content,
        }));
        if (history.length) setMessages(history);
      })
      .catch(() => {
        // History is a nice-to-have; a fresh empty conversation is a fine fallback.
      });
  }, [sessionId]);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeNode]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isSending) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setIsSending(true);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    send(sessionId, userId, text, {
      onFinal: (payload) => {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: payload.response,
            citations: payload.citations,
            escalated: payload.escalated,
            scopeLabel: payload.debug_scope_label,
            relevanceScore: payload.debug_relevance_score,
          },
        ]);
        setIsSending(false);
      },
      onError: () => {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Something went wrong reaching the server. Please try again.",
            scopeLabel: "off_topic", // reuse the muted rejected style for a plain error look
          },
        ]);
        setIsSending(false);
      },
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  };

  return (
    <div className="app-shell">
      <div className="chat-column">
        <header className="chat-header">
          <div className="brand-mark">
            <span className={`status-dot${isSending ? " busy" : ""}`} />
            <div>
              <div className="brand-name">Truelift Assistant</div>
              <div className="brand-subtitle">product &amp; support</div>
            </div>
          </div>
          <div className="header-actions">
            <button className="text-link-btn" onClick={onSwitchChat}>
              Switch chat
            </button>
            <span className="session-tag">session:{sessionId.slice(0, 8)}</span>
            <button className="icon-btn" onClick={onOpenAdmin} aria-label="Admin dashboard">
              <Settings size={16} />
            </button>
          </div>
        </header>

        <div className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Ask about Truelift</h2>
              <p>
                Pricing, onboarding, how incrementality measurement works, or
                how to reach the team -- ask anything about the product.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}

          {isSending && <PipelineStatus node={activeNode} />}

          <div ref={listEndRef} />
        </div>

        <div className="composer">
          <div className="composer-inner">
            <textarea
              ref={textareaRef}
              rows={1}
              placeholder="Message the assistant..."
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              disabled={isSending}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={isSending || !input.trim()}
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </div>
          <p className="composer-hint">Enter to send · Shift+Enter for a new line</p>
        </div>
      </div>
    </div>
  );
}

function LoadingSplash() {
  return (
    <div className="app-shell">
      <div className="chat-column">
        <div className="loading-splash">
          <span className="spinner" aria-hidden="true" />
          <span>Loading…</span>
        </div>
      </div>
    </div>
  );
}

function ChatApp() {
  const [userId] = useState(getOrCreateUserId);
  const [phase, setPhase] = useState("loading"); // loading | picker | chat
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [creating, setCreating] = useState(false);

  const startNewChat = useCallback(() => {
    setCreating(true);
    fetch(`${API_BASE_URL}/api/users/${userId}/sessions`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        setSessionId(data.session_id);
        setPhase("chat");
      })
      .catch(() => {
        // Backend unreachable -- fall back to a client-generated id so the
        // person can still use the chat UI; history just won't persist.
        setSessionId(crypto.randomUUID());
        setPhase("chat");
      })
      .finally(() => setCreating(false));
  }, [userId]);

  const loadSessions = useCallback(() => {
    setPhase("loading");
    fetch(`${API_BASE_URL}/api/users/${userId}/sessions`)
      .then((r) => r.json())
      .then((data) => {
        const list = data.sessions ?? [];
        if (list.length === 0) {
          startNewChat();
        } else {
          setSessions(list);
          setPhase("picker");
        }
      })
      .catch(() => {
        setSessionId(crypto.randomUUID());
        setPhase("chat");
      });
  }, [userId, startNewChat]);

  useEffect(() => {
    loadSessions();
    // Only run once on mount -- switching chats later is triggered
    // explicitly via onSwitchChat, not by this effect re-running.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (phase === "loading") return <LoadingSplash />;

  if (phase === "picker") {
    return (
      <SessionPicker
        sessions={sessions}
        creating={creating}
        onSelect={(id) => {
          setSessionId(id);
          setPhase("chat");
        }}
        onNew={startNewChat}
      />
    );
  }

  return (
    <ChatView
      userId={userId}
      sessionId={sessionId}
      onSwitchChat={loadSessions}
      onOpenAdmin={() => {
        window.location.hash = "admin";
      }}
    />
  );
}

export default function App() {
  const [route, setRoute] = useState(() => (window.location.hash === "#admin" ? "admin" : "app"));

  useEffect(() => {
    const onHashChange = () => setRoute(window.location.hash === "#admin" ? "admin" : "app");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  if (route === "admin") {
    return (
      <AdminPanel
        onBackToChat={() => {
          window.location.hash = "";
        }}
      />
    );
  }

  return <ChatApp />;
}
