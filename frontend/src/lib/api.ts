export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function failure(r: Response): Promise<ApiError> {
  let message = `Request failed (${r.status})`;
  try {
    const body = await r.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(r.status, message);
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const r = await fetch(`/api${path}`, {
    method: opts.method ?? (opts.body || opts.form ? "POST" : "GET"),
    headers: opts.body ? { "Content-Type": "application/json" } : undefined,
    body: opts.form ?? (opts.body ? JSON.stringify(opts.body) : undefined),
  });
  if (r.status === 401 && !path.startsWith("/auth") && typeof window !== "undefined") {
    window.location.href = "/login";
  }
  if (!r.ok) throw await failure(r);
  return r.json();
}

export type ChatEvent =
  | { event: "start"; conversation_id: string }
  | { event: "status"; stage: string }
  | { event: "token"; text: string }
  | { event: "replace"; text: string; reason?: string }
  | ({ event: "done" } & DoneMeta);

export type Source = { n: number; document_id: string; title: string; url: string; snippet: string; relevance: number };
export type DoneMeta = { message_id: string; status: string; verified: boolean | null; sources: Source[]; generator: string; ms: number };

export async function streamChat(
  body: { message: string; conversation_id?: string | null; channel?: "text" | "voice" },
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw await failure(r);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const line = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

export function ago(iso: string | null): string {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso + "Z").getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
