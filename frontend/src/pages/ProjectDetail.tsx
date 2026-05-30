import { useEffect, useState, useCallback } from 'react'
import { useNavigate, useParams, Link } from 'react-router-dom'
import { api, useProjectSSE } from '../api/client'
import type { ProjectRecord, ReviewSummary, SSEEvent } from '../types'
import StatusBadge from '../components/StatusBadge'

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

interface IndexingStep {
  id: string
  icon: string
  label: string
  detail?: string
  state: 'done' | 'running' | 'error'
}

function eventsToSteps(events: SSEEvent[]): IndexingStep[] {
  const steps: IndexingStep[] = []

  for (const ev of events) {
    const t = ev.type
    if (t === 'status') {
      const msg = String(ev.message ?? ev.msg ?? '')
      steps.push({
        id: `status-${steps.length}`,
        icon: 'ℹ',
        label: msg,
        state: 'done',
      })
    } else if (t === 'cloning' || t === 'clone') {
      const msg = String(ev.message ?? 'Cloning repository…')
      steps.push({
        id: `clone-${steps.length}`,
        icon: '⬇️',
        label: msg,
        state: 'running',
      })
    } else if (t === 'progress' || t === 'indexing') {
      const pct = Number(ev.progress_pct ?? ev.pct ?? 0)
      const msg = String(ev.message ?? 'Indexing files…')
      steps.push({
        id: `prog-${steps.length}`,
        icon: '⚙️',
        label: msg,
        detail: pct ? `${Math.round(pct)}%` : undefined,
        state: 'running',
      })
    } else if (t === 'file' || t === 'file_indexed') {
      const path = String(ev.file_path ?? ev.path ?? '')
      const current = Number(ev.current ?? 0)
      const total = Number(ev.total ?? 0)
      steps.push({
        id: `file-${steps.length}`,
        icon: '📄',
        label: `Indexed ${path}`,
        detail: total ? `(${current}/${total})` : undefined,
        state: 'done',
      })
    } else if (t === 'done') {
      const nodes = Number(ev.node_count ?? ev.nodes ?? 0)
      const files = Number(ev.file_count ?? ev.files ?? 0)
      steps.push({
        id: `done-${steps.length}`,
        icon: '✅',
        label: 'Indexing complete!',
        detail: nodes || files ? `${files} files, ${nodes} nodes` : undefined,
        state: 'done',
      })
    } else if (t === 'error') {
      const msg = String(ev.message ?? ev.error ?? 'Unknown error')
      steps.push({
        id: `err-${steps.length}`,
        icon: '❌',
        label: `Error: ${msg}`,
        state: 'error',
      })
    } else if (t === 'warning') {
      const msg = String(ev.message ?? ev.warning ?? '')
      steps.push({
        id: `warn-${steps.length}`,
        icon: '⚠️',
        label: msg,
        state: 'done',
      })
    } else {
      // Generic fallback: show message if present
      const msg = String(ev.message ?? '')
      if (msg) {
        steps.push({
          id: `gen-${steps.length}`,
          icon: '•',
          label: msg,
          state: 'done',
        })
      }
    }
  }

  return steps
}

function IndexingTimeline({ events, connected }: { events: SSEEvent[]; connected: boolean }) {
  const steps = eventsToSteps(events)

  if (steps.length === 0 && !connected) {
    return (
      <div className="text-sm text-gray-400 italic py-4 text-center">
        Waiting for indexing events...
      </div>
    )
  }

  return (
    <div className="relative pl-6">
      <div className="absolute left-3 top-2 bottom-2 w-0.5 bg-gray-200" />

      <div className="space-y-3">
        {steps.map((step, idx) => {
          const isLast = idx === steps.length - 1
          return (
            <div key={step.id} className="relative flex items-start gap-3">
              <div
                className={`absolute -left-3 w-3 h-3 rounded-full border-2 mt-0.5 ${
                  step.state === 'error'
                    ? 'bg-red-500 border-red-400'
                    : step.state === 'running' && isLast && connected
                    ? 'bg-blue-500 border-blue-400 animate-pulse'
                    : step.state === 'running'
                    ? 'bg-blue-500 border-blue-400'
                    : 'bg-green-500 border-green-400'
                }`}
              />

              <div
                className={`ml-2 flex-1 rounded-lg px-3 py-2 text-sm ${
                  step.state === 'error'
                    ? 'bg-red-50 border border-red-200'
                    : step.state === 'running'
                    ? 'bg-blue-50 border border-blue-200'
                    : 'bg-white border border-gray-100'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span>{step.icon}</span>
                  <span
                    className={`font-medium ${
                      step.state === 'error' ? 'text-red-700' : 'text-gray-800'
                    }`}
                  >
                    {step.label}
                  </span>
                </div>
                {step.detail && (
                  <div className="mt-0.5 ml-6 text-xs text-gray-500 truncate max-w-lg">
                    {step.detail}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [project, setProject] = useState<ProjectRecord | null>(null)
  const [reviews, setReviews] = useState<ReviewSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const isActive =
    project?.status === 'pending' ||
    project?.status === 'cloning' ||
    project?.status === 'indexing'

  const { events, connected } = useProjectSSE(id, isActive)

  const fetchProject = useCallback(async () => {
    if (!id) return
    try {
      const data = await api.getProject(id)
      setProject(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load project')
    } finally {
      setLoading(false)
    }
  }, [id])

  const fetchReviews = useCallback(async () => {
    if (!id) return
    try {
      const data = await api.listProjectReviews(id)
      setReviews(data)
    } catch {
      // non-fatal — reviews list not critical
    }
  }, [id])

  useEffect(() => {
    fetchProject()
    fetchReviews()
  }, [fetchProject, fetchReviews])

  // Poll while indexing as a fallback
  useEffect(() => {
    if (!isActive) return
    const interval = setInterval(() => {
      fetchProject()
    }, 5_000)
    return () => clearInterval(interval)
  }, [isActive, fetchProject])

  // Re-fetch once SSE closes (indexing finished or errored)
  useEffect(() => {
    if (!connected && events.length > 0) {
      fetchProject()
      fetchReviews()
    }
  }, [connected, events.length, fetchProject, fetchReviews])

  const handleReindex = async () => {
    if (!id) return
    if (!confirm('Re-index this project? This will re-clone and rebuild the graph.')) return
    try {
      setActionError(null)
      const updated = await api.reindexProject(id)
      setProject(updated)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to re-index project')
    }
  }

  const handleDelete = async () => {
    if (!id || !project) return
    if (!confirm(`Delete project "${project.repo_name}"? This will remove its graph data.`)) return
    try {
      setActionError(null)
      await api.deleteProject(id)
      navigate('/projects')
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to delete project')
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-500">
          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading project...
        </div>
      </div>
    )
  }

  if (error || !project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 font-medium mb-2">{error ?? 'Project not found'}</p>
          <Link to="/projects" className="text-blue-600 hover:underline text-sm">
            ← Back to projects
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-3">
          <Link to="/projects" className="text-gray-400 hover:text-gray-600 transition-colors text-sm">
            ← Projects
          </Link>
          <span className="text-gray-300">|</span>
          <span className="text-2xl">🗂️</span>
          <h1 className="text-xl font-bold text-gray-900 truncate">{project.repo_name}</h1>
          <StatusBadge status={project.status} />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        {/* Section A: Status & Stats */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
            <div className="space-y-2 min-w-0">
              <a
                href={project.repo_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 hover:underline text-sm font-mono break-all"
              >
                {project.repo_url} ↗
              </a>
              <div className="text-xs text-gray-400">
                Default branch: <span className="font-mono">{project.default_branch}</span>
              </div>
              <div className="text-xs text-gray-400">
                Created {formatDate(project.created_at)}
              </div>
            </div>
          </div>

          {project.error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              <strong>Error:</strong> {project.error}
            </div>
          )}

          {project.status !== 'indexed' ? (
            <div>
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm text-gray-600 mb-1.5">
                  <span className="font-medium">
                    {project.status === 'cloning'
                      ? 'Cloning repository…'
                      : project.status === 'indexing'
                      ? 'Building knowledge graph…'
                      : project.status === 'pending'
                      ? 'Queued…'
                      : project.status === 'failed'
                      ? 'Indexing failed'
                      : 'Working…'}
                  </span>
                  <span className="text-xs text-gray-500">
                    {Math.round(project.progress_pct)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      project.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{
                      width: `${Math.min(100, Math.max(0, project.progress_pct))}%`,
                    }}
                  />
                </div>
              </div>

              {isActive && (
                <div className="mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <h3 className="text-sm font-semibold text-gray-800">Indexing Progress</h3>
                    {connected && (
                      <span className="flex items-center gap-1.5 text-xs text-blue-600 font-medium">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                        Live
                      </span>
                    )}
                  </div>
                  <IndexingTimeline events={events} connected={connected} />
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-2">
              <Stat label="Files" value={project.file_count.toLocaleString()} />
              <Stat label="Nodes" value={project.node_count.toLocaleString()} />
              <Stat label="Edges" value={project.edge_count.toLocaleString()} />
              <Stat
                label="Last commit"
                value={
                  project.last_commit_sha
                    ? project.last_commit_sha.slice(0, 7)
                    : '—'
                }
                mono
              />
              <Stat label="Last indexed" value={formatDate(project.last_indexed_at)} />
            </div>
          )}
        </div>

        {/* Section B: Actions */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-800 mb-3">Actions</h3>
          {actionError && (
            <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              {actionError}
            </div>
          )}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={handleReindex}
              disabled={isActive}
              className="px-4 py-2 bg-blue-50 hover:bg-blue-100 disabled:bg-gray-50 disabled:text-gray-400 text-blue-700 text-sm font-medium rounded-lg transition-colors"
            >
              Re-index
            </button>
            <button
              onClick={handleDelete}
              className="px-4 py-2 bg-red-50 hover:bg-red-100 text-red-700 text-sm font-medium rounded-lg transition-colors ml-auto"
            >
              Delete project
            </button>
          </div>
        </div>

        {/* Section C: Reviews of this project */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-800">Reviews for this project</h3>
          </div>
          {reviews.length === 0 ? (
            <div className="px-6 py-10 text-center text-gray-400">
              <p className="text-sm">No reviews yet for this project.</p>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    PR #
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    Patches
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    Created
                  </th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {reviews.map((r) => (
                  <tr
                    key={r.id}
                    className="border-t border-gray-100 hover:bg-gray-50 transition-colors"
                  >
                    <td className="px-4 py-3 font-medium text-gray-900">
                      #{r.pr_number}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      {r.total_patches ?? '—'}
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">
                      {formatDate(r.created_at)}
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        to={`/review/${r.id}`}
                        className="inline-flex items-center gap-1 px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-medium rounded-md transition-colors"
                      >
                        View →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  )
}

function Stat({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-gray-400 mb-0.5">{label}</div>
      <div
        className={`text-base font-semibold text-gray-900 ${mono ? 'font-mono' : ''}`}
      >
        {value}
      </div>
    </div>
  )
}
