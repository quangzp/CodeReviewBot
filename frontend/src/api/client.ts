import { useEffect, useRef, useState } from 'react'
import type {
  ChatSessionFull,
  ChatSessionSummary,
  ProjectRecord,
  ProjectSummary,
  ReviewRecord,
  ReviewSummary,
  SSEEvent,
} from '../types'

const BASE_URL = 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

// ============================================================================
// Developer profile types
// ============================================================================
export interface DeveloperSummary {
  login: string
  name?: string
  avatar_url?: string
  first_seen_at: string
  last_seen_at: string
  pr_count: number
}

export interface BugPatternData {
  id: string
  name: string
  description: string
  severity: string
  occurrence_count: number
}

export interface PatternWithEvidence {
  pattern: BugPatternData
  confidence: number
  evidence_count: number
  last_observed_at: string
  is_active: boolean
}

export interface DeveloperProfileData {
  profile: {
    developer: DeveloperSummary
    patterns: PatternWithEvidence[]
  }
  recent_reviews: Array<{
    id: string
    pr_url: string
    pr_number: number
    repo_name: string
    summary: string
    reviewed_at: string
    patterns: string[]
  }>
}

export const api = {
  createReview: (pr_url: string): Promise<ReviewRecord> =>
    request<ReviewRecord>('/api/reviews', {
      method: 'POST',
      body: JSON.stringify({ pr_url }),
    }),

  listReviews: (): Promise<ReviewSummary[]> =>
    request<ReviewSummary[]>('/api/reviews'),

  getReview: (id: string): Promise<ReviewRecord> =>
    request<ReviewRecord>(`/api/reviews/${id}`),

  listDevelopers: (): Promise<DeveloperSummary[]> =>
    request<DeveloperSummary[]>('/api/developers'),

  getDeveloper: (login: string): Promise<DeveloperProfileData> =>
    request<DeveloperProfileData>(`/api/developers/${login}`),

  createProject: (repo_url: string, default_branch?: string): Promise<ProjectRecord> =>
    request<ProjectRecord>('/api/projects', {
      method: 'POST',
      body: JSON.stringify(
        default_branch ? { repo_url, default_branch } : { repo_url }
      ),
    }),

  listProjects: (): Promise<ProjectSummary[]> =>
    request<ProjectSummary[]>('/api/projects'),

  getProject: (id: string): Promise<ProjectRecord> =>
    request<ProjectRecord>(`/api/projects/${id}`),

  reindexProject: (id: string): Promise<ProjectRecord> =>
    request<ProjectRecord>(`/api/projects/${id}/reindex`, { method: 'POST' }),

  deleteProject: (id: string): Promise<void> =>
    fetch(`${BASE_URL}/api/projects/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
    }),

  listProjectReviews: (id: string): Promise<ReviewSummary[]> =>
    request<ReviewSummary[]>(`/api/projects/${id}/reviews`),

  listSessions: (): Promise<ChatSessionSummary[]> =>
    request<ChatSessionSummary[]>('/api/chat/sessions'),

  createSession: (title?: string): Promise<ChatSessionFull> =>
    request<ChatSessionFull>('/api/chat/sessions', {
      method: 'POST',
      body: JSON.stringify(title ? { title } : {}),
    }),

  getSession: (id: string): Promise<ChatSessionFull> =>
    request<ChatSessionFull>(`/api/chat/sessions/${id}`),

  renameSession: (id: string, title: string): Promise<void> =>
    fetch(`${BASE_URL}/api/chat/sessions/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ title }),
    }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
    }),

  deleteSession: (id: string): Promise<void> =>
    fetch(`${BASE_URL}/api/chat/sessions/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
    }),
}

// ============================================================================
// Chat / Agent types
// ============================================================================
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export type ToolRender =
  | { kind: 'project'; project: ProjectRecord; already_exists?: boolean; indexing?: boolean }
  | { kind: 'project_list'; projects: ProjectSummary[] }
  | { kind: 'review'; review_id: string; pr_url: string; repo_name: string; pr_number: number }
  | { kind: 'review_list'; reviews: ReviewSummary[] }
  | {
      kind: 'fix_attempt'
      status: 'success' | 'no_file' | 'no_content' | 'patch_failed'
      file_path?: string
      fault_description?: string
      patch?: string
      attempts?: number
      applies_cleanly?: boolean
      repo_name?: string
      last_error?: string
      bug_description?: string
    }
  | { kind: 'error'; message: string; suggestion?: string }

export type AgentEvent =
  | { type: 'session'; id: string; title: string }
  | { type: 'thinking'; step: number }
  | { type: 'tool_call'; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; name: string; summary: string; render: ToolRender }
  | { type: 'message'; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string }

export function streamChat(
  message: string,
  history: ChatMessage[],
  onEvent: (event: AgentEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
  sessionId?: string,
): { cancel: () => void } {
  const controller = new AbortController()

  ;(async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        credentials: 'include',
        body: JSON.stringify(
          sessionId ? { message, history, session_id: sessionId } : { message, history },
        ),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => '')
        onError(text || `HTTP ${res.status}`)
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE messages separated by blank lines
        let sepIdx: number
        while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
          const chunk = buffer.slice(0, sepIdx)
          buffer = buffer.slice(sepIdx + 2)

          // Extract data: lines, join them
          const dataLines = chunk
            .split('\n')
            .filter((l) => l.startsWith('data:'))
            .map((l) => l.slice(5).trimStart())
          if (dataLines.length === 0) continue
          const data = dataLines.join('\n')
          if (!data) continue
          try {
            const parsed = JSON.parse(data) as AgentEvent
            onEvent(parsed)
            if (parsed.type === 'done') {
              onDone()
              return
            }
            if (parsed.type === 'error') {
              onError(parsed.message)
              return
            }
          } catch {
            // ignore malformed
          }
        }
      }

      onDone()
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      onError(err instanceof Error ? err.message : String(err))
    }
  })()

  return { cancel: () => controller.abort() }
}

// SSE hook — connects to /api/reviews/:id/stream
export function useSSE(id: string | undefined, enabled: boolean) {
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!id || !enabled) return

    const es = new EventSource(`${BASE_URL}/api/reviews/${id}/stream`)
    esRef.current = es
    setConnected(true)

    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(e.data) as SSEEvent
        setEvents((prev) => [...prev, parsed])
        if (parsed.type === 'done' || parsed.type === 'error') {
          es.close()
          setConnected(false)
        }
      } catch {
        // ignore malformed events
      }
    }

    es.onerror = () => {
      es.close()
      setConnected(false)
    }

    return () => {
      es.close()
      setConnected(false)
    }
  }, [id, enabled])

  return { events, connected }
}

// SSE hook — connects to /api/projects/:id/stream for indexing progress
export function useProjectSSE(id: string | undefined, enabled: boolean) {
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!id || !enabled) return

    const es = new EventSource(`${BASE_URL}/api/projects/${id}/stream`)
    esRef.current = es
    setConnected(true)

    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(e.data) as SSEEvent
        setEvents((prev) => [...prev, parsed])
        if (parsed.type === 'done' || parsed.type === 'error') {
          es.close()
          setConnected(false)
        }
      } catch {
        // ignore malformed events
      }
    }

    es.onerror = () => {
      es.close()
      setConnected(false)
    }

    return () => {
      es.close()
      setConnected(false)
    }
  }, [id, enabled])

  return { events, connected }
}
