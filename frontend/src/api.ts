export type Citation = {
  label: string
  doc_id: string
  title: string
  section: string
  page: number | null
  score: number
  snippet: string
}

export type Source = { label: string; doc_id: string; title: string; section: string; score: number }

export type AgentEvent =
  | { type: 'start'; trace_id: string }
  | { type: 'tool_call'; id: string; name: string; arguments: Record<string, unknown>; step: number }
  | {
      type: 'tool_result'
      id: string
      name: string
      error: boolean
      data: { sources?: Source[]; rows?: unknown[][]; columns?: string[]; result?: number; sql?: string }
    }
  | { type: 'token'; text: string }
  | { type: 'discard_text' }
  | {
      type: 'done'
      trace_id: string
      answer: string
      citations: Citation[]
      steps: number
      latency_ms: number
      usage: { input_tokens: number; output_tokens: number }
      cost_usd: number | null
      provider: string
      model: string
    }
  | { type: 'error'; message: string }

export type Span = {
  trace_id: string
  span_id: string
  parent_id: string | null
  name: string
  start_ns: number
  duration_ms: number
  status: string
  attributes: Record<string, unknown>
  events: { name: string; attributes: Record<string, unknown> }[]
}

export type Health = {
  status: string
  llm: { provider: string; model: string } | null
  llm_error: string | null
  embedding_model: string
  indexed_chunks: number
}

const API = import.meta.env.VITE_API_URL ?? ''

/** POST a question and yield Server-Sent Events as they arrive. */
export async function* ask(question: string, signal?: AbortSignal): AsyncGenerator<AgentEvent> {
  const response = await fetch(`${API}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!response.ok || !response.body) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail ?? `Request failed (${response.status})`)
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value.replace(/\r\n/g, '\n')
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (data) yield JSON.parse(data) as AgentEvent
    }
  }
}

export async function getTrace(traceId: string): Promise<Span[]> {
  const response = await fetch(`${API}/api/traces/${traceId}`)
  if (!response.ok) throw new Error(`Trace not found (${response.status})`)
  return response.json()
}

export async function getHealth(): Promise<Health> {
  const response = await fetch(`${API}/api/health`)
  return response.json()
}
