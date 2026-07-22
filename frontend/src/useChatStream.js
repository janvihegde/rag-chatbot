import { useCallback, useRef, useState } from "react";
import { API_BASE_URL } from "./config";

// The backend's pipeline node names, in the order the graph actually runs
// them (see backend/app/graph.py). Used to render a live "what's happening
// right now" readout while a response streams in.
export const PIPELINE_STEPS = {
  scope_check: "Checking scope",
  retrieval: "Searching documents",
  relevance_gate: "Checking relevance",
  generation: "Writing answer",
  escalation: "Escalating to a human",
  scope_reject: "Declining out-of-scope request",
};

/**
 * Consumes POST /api/chat's SSE stream. EventSource can't send a POST body,
 * so this reads the raw response stream and parses "data: {...}\n\n" frames
 * by hand -- same event format, just done manually.
 */
export function useChatStream() {
  const [activeNode, setActiveNode] = useState(null);
  const abortRef = useRef(null);

  const send = useCallback(async (sessionId, userId, message, { onNodeUpdate, onFinal, onError }) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_id: userId, message }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? ""; // last chunk may be incomplete, keep it for next read

        for (const frame of frames) {
          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const payload = JSON.parse(line.slice("data: ".length));

          if (payload.event === "node_update") {
            setActiveNode(payload.node);
            onNodeUpdate?.(payload.node);
          } else if (payload.event === "final_result") {
            setActiveNode(null);
            onFinal?.(payload);
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setActiveNode(null);
        onError?.(err);
      }
    }
  }, []);

  return { send, activeNode };
}