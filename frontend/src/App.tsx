import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ask, getHealth, getTrace, type AgentEvent, type Citation, type Health, type Span } from './api'

const EXAMPLES = [
  'What happens when zip() is called with strict=True and the iterables have different lengths?',
  'Which PEPs in the corpus target Python 3.12?',
  'PEP 8 sets one line length for code and a lower one for comments. What is the difference?',
  'How do I build CPython without the GIL?',
]

type ToolStep = { id: string; name: string; args: Record<string, unknown>; result?: Extract<AgentEvent, { type: 'tool_result' }> }
type Done = Extract<AgentEvent, { type: 'done' }>

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [steps, setSteps] = useState<ToolStep[]>([])
  const [done, setDone] = useState<Done | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [spans, setSpans] = useState<Span[] | null>(null)
  const [activeCitation, setActiveCitation] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  async function submit(q: string) {
    if (!q.trim() || busy) return
    abort.current?.abort()
    const controller = new AbortController()
    abort.current = controller
    setQuestion(q)
    setAnswer('')
    setSteps([])
    setDone(null)
    setError(null)
    setSpans(null)
    setActiveCitation(null)
    setBusy(true)
    try {
      for await (const event of ask(q, controller.signal)) {
        switch (event.type) {
          case 'token':
            setAnswer((a) => a + event.text)
            break
          case 'discard_text':
            setAnswer('')
            break
          case 'tool_call':
            setSteps((s) => [...s, { id: event.id, name: event.name, args: event.arguments }])
            break
          case 'tool_result':
            setSteps((s) => s.map((step) => (step.id === event.id ? { ...step, result: event } : step)))
            break
          case 'done':
            setAnswer(event.answer)
            setDone(event)
            getTrace(event.trace_id).then(setSpans).catch(() => setSpans(null))
            break
          case 'error':
            setError(event.message)
            break
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const citations = new Map((done?.citations ?? []).map((c) => [c.label, c]))

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Doc QA agent</h1>
          <p className="muted">
            Ask about 25 Python Enhancement Proposals. The agent searches a vector index, queries document metadata
            with SQL, uses a calculator, and cites its sources.
          </p>
        </div>
        <HealthBadge health={health} />
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault()
          submit(question)
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the documents…"
          aria-label="Question"
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? 'Thinking…' : 'Ask'}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => submit(ex)} disabled={busy}>
            {ex}
          </button>
        ))}
      </div>

      {(steps.length > 0 || answer || error || busy) && (
        <main className="grid">
          <section className="card answer">
            <h2>Answer</h2>
            {error && <p className="error">{error}</p>}
            {!answer && busy && <p className="muted">Working…</p>}
            <AnswerText text={answer} citations={citations} onCite={setActiveCitation} active={activeCitation} />
            {done && <Metrics done={done} />}
            {done && done.citations.length > 0 && (
              <div className="sources">
                <h3>Sources</h3>
                {done.citations.map((c) => (
                  <SourceCard key={c.label} citation={c} active={activeCitation === c.label} onClick={() => setActiveCitation(c.label)} />
                ))}
              </div>
            )}
          </section>

          <aside className="card">
            <h2>Agent steps</h2>
            {steps.length === 0 && <p className="muted">No tool calls yet.</p>}
            <ol className="steps">
              {steps.map((step) => (
                <li key={step.id}>
                  <span className="tool">{step.name}</span>
                  <code className="args">{formatArgs(step.args)}</code>
                  {step.result && <ToolSummary result={step.result} />}
                </li>
              ))}
            </ol>
            {spans && <TraceView spans={spans} />}
          </aside>
        </main>
      )}
    </div>
  )
}

function HealthBadge({ health }: { health: Health | null }) {
  if (!health) return <span className="badge bad">API offline</span>
  if (!health.llm)
    return (
      <span className="badge bad" title={health.llm_error ?? ''}>
        LLM not configured
      </span>
    )
  return (
    <span className="badge" title={`${health.indexed_chunks} chunks, embeddings ${health.embedding_model}`}>
      {health.llm.provider} · {health.llm.model}
    </span>
  )
}

function formatArgs(args: Record<string, unknown>) {
  const values = Object.values(args)
  return values.length === 1 ? String(values[0]) : JSON.stringify(args)
}

function ToolSummary({ result }: { result: Extract<AgentEvent, { type: 'tool_result' }> }) {
  if (result.error) return <div className="result error">error</div>
  const { data } = result
  if (data.sources)
    return (
      <div className="result">
        {data.sources.map((s) => (
          <div key={s.label}>
            <b>[{s.label}]</b> {s.title.replace(/^PEP \d+: /, '')} <span className="muted">{s.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    )
  if (data.rows)
    return (
      <div className="result">
        {data.rows.length} row{data.rows.length === 1 ? '' : 's'}:{' '}
        {data.rows
          .slice(0, 4)
          .map((r) => r.join(', '))
          .join(' · ')}
      </div>
    )
  if (data.result !== undefined) return <div className="result">= {data.result}</div>
  return null
}

/** Minimal rendering: paragraphs, bullet lists, `code`, **bold**, and [S1] citation chips. */
function AnswerText({
  text,
  citations,
  onCite,
  active,
}: {
  text: string
  citations: Map<string, Citation>
  onCite: (label: string) => void
  active: string | null
}) {
  const inline = (line: string, key: string): ReactNode[] =>
    line.split(/(\[S\d+(?:,\s*S\d+)*\]|`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
      const k = `${key}-${i}`
      if (/^\[S\d+/.test(part))
        return part
          .slice(1, -1)
          .split(/,\s*/)
          .map((label) => (
            <button
              key={`${k}-${label}`}
              className={`cite ${citations.has(label) ? '' : 'unknown'} ${active === label ? 'active' : ''}`}
              onClick={() => onCite(label)}
              title={citations.get(label)?.title}
            >
              {label}
            </button>
          ))
      if (part.startsWith('`')) return <code key={k}>{part.slice(1, -1)}</code>
      if (part.startsWith('**')) return <strong key={k}>{part.slice(2, -2)}</strong>
      return part
    })

  const blocks = text.split(/\n{2,}/).filter((b) => b.trim())
  return (
    <div className="answer-text">
      {blocks.map((block, b) => {
        const lines = block.split('\n')
        if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l)))
          return (
            <ul key={b}>
              {lines.map((l, i) => (
                <li key={i}>{inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ''), `${b}-${i}`)}</li>
              ))}
            </ul>
          )
        return <p key={b}>{lines.flatMap((l, i) => [...(i ? [<br key={`br${i}`} />] : []), ...inline(l, `${b}-${i}`)])}</p>
      })}
    </div>
  )
}

function SourceCard({ citation, active, onClick }: { citation: Citation; active: boolean; onClick: () => void }) {
  return (
    <button className={`source ${active ? 'active' : ''}`} onClick={onClick}>
      <div>
        <b>[{citation.label}]</b> {citation.title}
        {citation.section && <span className="muted"> › {citation.section}</span>}
      </div>
      {active && <pre className="snippet">{citation.snippet}</pre>}
    </button>
  )
}

function Metrics({ done }: { done: Done }) {
  const items = [
    ['Latency', `${(done.latency_ms / 1000).toFixed(2)} s`],
    ['Steps', String(done.steps)],
    ['Tokens', `${done.usage.input_tokens.toLocaleString()} in / ${done.usage.output_tokens.toLocaleString()} out`],
    ['Est. cost', done.cost_usd == null ? 'n/a' : `$${done.cost_usd.toFixed(5)}`],
  ]
  return (
    <dl className="metrics">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Waterfall of the request's OpenTelemetry spans, indented by depth. */
function TraceView({ spans }: { spans: Span[] }) {
  const root = spans.find((s) => s.parent_id === null)
  if (!root) return null
  const depth = new Map<string, number>([[root.span_id, 0]])
  const ordered: Span[] = []
  const visit = (span: Span) => {
    ordered.push(span)
    spans
      .filter((s) => s.parent_id === span.span_id)
      .sort((a, b) => a.start_ns - b.start_ns)
      .forEach((child) => {
        depth.set(child.span_id, (depth.get(span.span_id) ?? 0) + 1)
        visit(child)
      })
  }
  visit(root)
  const total = root.duration_ms || 1
  return (
    <div className="trace">
      <h3>
        Trace <code className="muted">{root.trace_id.slice(0, 12)}</code>
      </h3>
      {ordered.map((span) => {
        const offset = (span.start_ns - root.start_ns) / 1e6
        const a = span.attributes
        const detail = [
          span.name === 'llm.chat' ? `${a['gen_ai.usage.input_tokens']}→${a['gen_ai.usage.output_tokens']} tok` : null,
          a['retrieval.top_k'] !== undefined ? `k=${a['retrieval.top_k']}` : null,
        ]
          .filter(Boolean)
          .join(' ')
        return (
          <div className="span-row" key={span.span_id} title={JSON.stringify(span.attributes, null, 2)}>
            <div className="span-name" style={{ paddingLeft: `${(depth.get(span.span_id) ?? 0) * 12}px` }}>
              {span.name} <span className="muted">{detail}</span>
            </div>
            <div className="span-bar-track">
              <div
                className={`span-bar ${span.status === 'ERROR' ? 'err' : ''} kind-${span.name.split('.')[0]}`}
                style={{ left: `${(offset / total) * 100}%`, width: `${Math.max((span.duration_ms / total) * 100, 0.5)}%` }}
              />
            </div>
            <div className="span-ms">{span.duration_ms.toFixed(0)} ms</div>
          </div>
        )
      })}
    </div>
  )
}
