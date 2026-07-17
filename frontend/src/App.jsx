import { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { API_BASE_URL } from "./config";
import { useChatStream, PIPELINE_STEPS } from "./useChatStream";

function getOrCreateSessionId() {
  const key = "truelift_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

function ConfidenceChip({ score, source }) {
  const pct = Math.round((score ?? 0) * 100);
  const filledTicks = Math.round((score ?? 0) * 5);
  return (
    <div className="confidence-chip">
      <span>{pct}% match</span>
      <div className="ticks" aria-hidden="true">
        {Array.from({ length: 5 }).map((_, i) => (
          <span key={i} className={`tick${i < filledTicks ? " filled" : ""}`} />
        ))}
      </div>
      {source && (
        <>
          <span className="divider">·</span>
          <span className="source">{source}</span>
        </>
      )}
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  const isEscalated = !isUser && msg.escalated;
  const isRejected = !isUser && (msg.scopeLabel === "off_topic" || msg.scopeLabel === "injection");
  const hasConfidence = !isUser && !isEscalated && !isRejected && msg.citations?.length > 0;

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
          {msg.content}
        </div>
        {isEscalated && (
          <div className="escalate-tag">
            <span>↗</span> Escalated to the team
          </div>
        )}
        {hasConfidence && (
          <ConfidenceChip score={msg.relevanceScore} source={msg.citations[0]} />
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

export default function App() {
  const [sessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const { send, activeNode } = useChatStream();
  const listEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Load prior history for this session on first mount (so a refresh
  // doesn't lose the conversation -- the backend already persists it).
  useEffect(() => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

    send(sessionId, text, {
      onFinal: (payload) => {
        console.log("Final payload:", payload);
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
          <span className="session-tag">session:{sessionId.slice(0, 8)}</span>
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