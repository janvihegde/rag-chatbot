import { useEffect, useState } from "react";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { fetchAnalytics, fetchEscalations } from "./adminApi";

const TOKEN_KEY = "truelift_admin_token";

function formatTimestamp(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// Interpolates between --lift (0% escalation, best case) and --escalate
// (high escalation, worst case) so the gauge's color itself communicates
// health at a glance, using the same two semantic colors the chat UI
// already uses for "answered" vs "escalated".
function liftColor(escalationRatePercent) {
  const t = Math.min(Math.max(escalationRatePercent / 50, 0), 1); // 50%+ reads as "worst"
  const good = [22, 163, 116]; // --lift
  const bad = [181, 84, 12]; // --escalate
  const mix = good.map((c, i) => Math.round(c + (bad[i] - c) * t));
  return `rgb(${mix.join(",")})`;
}

function LiftGauge({ escalationRatePercent }) {
  const pct = Math.min(Math.max(escalationRatePercent, 0), 100);
  const color = liftColor(pct);
  // Semicircle gauge built as a single SVG arc -- avoids pulling in a
  // charting library for one shape, and keeps exact control over the
  // "answered vs escalated" framing that's specific to this product.
  const radius = 84;
  const circumference = Math.PI * radius; // half-circle arc length
  const filled = (pct / 100) * circumference;

  return (
    <div className="lift-gauge">
      <svg viewBox="0 0 200 116" className="lift-gauge-svg">
        <path
          d="M 16 100 A 84 84 0 0 1 184 100"
          fill="none"
          stroke="var(--border)"
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M 16 100 A 84 84 0 0 1 184 100"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
        />
      </svg>
      <div className="lift-gauge-readout">
        <div className="lift-gauge-value" style={{ color }}>
          {pct.toFixed(1)}%
        </div>
        <div className="lift-gauge-label">of queries escalated</div>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
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
      await fetchAnalytics(value.trim());
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
        <p>Enter the admin token to view escalations and analytics.</p>
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

export default function AdminPanel({ onBackToChat }) {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [analytics, setAnalytics] = useState(null);
  const [escalations, setEscalations] = useState([]);
  const [statusFilter, setStatusFilter] = useState("pending");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadData = async (currentToken) => {
    setLoading(true);
    setError(null);
    try {
      const [analyticsData, escalationsData] = await Promise.all([
        fetchAnalytics(currentToken),
        fetchEscalations(currentToken, statusFilter),
      ]);
      setAnalytics(analyticsData.metrics);
      setEscalations(escalationsData.escalations ?? []);
    } catch (err) {
      if (err.isAuthError) {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
      } else {
        setError("Couldn't load dashboard data. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) loadData(token);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, statusFilter]);

  if (!token) {
    return <TokenGate onUnlock={setToken} />;
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
              <div className="brand-subtitle">escalations &amp; analytics</div>
            </div>
          </div>
          <button
            className="icon-btn"
            onClick={() => loadData(token)}
            aria-label="Refresh"
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? "spin" : ""} />
          </button>
        </header>

        <div className="admin-body">
          {error && <div className="admin-error-banner">{error}</div>}

          {analytics && (
            <section className="gauge-section">
              <LiftGauge escalationRatePercent={analytics.escalation_rate_percent} />
              <div className="stat-row">
                <StatCard label="Total queries" value={analytics.total_user_queries} />
                <StatCard label="Total escalations" value={analytics.total_escalations} />
                <StatCard label="Pending" value={analytics.pending_escalations} />
              </div>
            </section>
          )}

          <section className="escalation-section">
            <div className="escalation-header">
              <h2>Escalation queue</h2>
              <div className="status-tabs">
                {["pending", "resolved"].map((s) => (
                  <button
                    key={s}
                    className={`status-tab${statusFilter === s ? " active" : ""}`}
                    onClick={() => setStatusFilter(s)}
                  >
                    {s === "pending" ? "Pending" : "Resolved"}
                  </button>
                ))}
              </div>
            </div>

            {escalations.length === 0 && !loading && (
              <div className="empty-state small">
                <p>No {statusFilter} escalations right now.</p>
              </div>
            )}

            <div className="escalation-list">
              {escalations.map((esc) => (
                <div key={esc._id} className="escalation-row">
                  <div className="escalation-main">
                    <div className="escalation-message">{esc.user_message}</div>
                    <div className="escalation-meta">
                      <span className="session-tag">
                        session:{esc.session_id?.slice(0, 8)}
                      </span>
                      <span className="divider">·</span>
                      <span>{formatTimestamp(esc.timestamp)}</span>
                      <span className="divider">·</span>
                      <span>relevance {(esc.relevance_score ?? 0).toFixed(4)}</span>
                    </div>
                  </div>
                  <span className={`status-pill ${esc.status}`}>{esc.status}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}