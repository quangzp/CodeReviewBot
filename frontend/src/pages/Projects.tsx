import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ProjectSummary } from '../types'
import StatusBadge from '../components/StatusBadge'

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const navigate = useNavigate()

  const fetchProjects = async () => {
    try {
      const data = await api.listProjects()
      setProjects(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load projects')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchProjects()
    const interval = setInterval(fetchProjects, 5_000)
    return () => clearInterval(interval)
  }, [])

  const handleReindex = async (id: string) => {
    if (!confirm('Re-index this project? This will re-clone and rebuild the graph.')) return
    try {
      setActionError(null)
      await api.reindexProject(id)
      fetchProjects()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to re-index project')
    }
  }

  const handleDelete = async (id: string, repoName: string) => {
    if (!confirm(`Delete project "${repoName}"? This will remove its graph data.`)) return
    try {
      setActionError(null)
      await api.deleteProject(id)
      fetchProjects()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to delete project')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="text-gray-400 hover:text-gray-600 transition-colors text-sm">
              ← Back to dashboard
            </Link>
            <span className="text-gray-300">|</span>
            <span className="text-2xl">🗂️</span>
            <h1 className="text-xl font-bold text-gray-900">Projects</h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/projects/new')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
            >
              <span className="text-base leading-none">+</span>
              Add Project
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Indexed Projects</h2>
          <span className="text-xs text-gray-400">Auto-refreshes every 5s</span>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        {actionError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {actionError}
          </div>
        )}

        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="bg-white rounded-2xl shadow-sm border border-gray-200 p-5 animate-pulse"
              >
                <div className="h-5 bg-gray-200 rounded w-2/3 mb-3" />
                <div className="h-4 bg-gray-100 rounded w-1/3 mb-4" />
                <div className="h-3 bg-gray-100 rounded w-1/2" />
              </div>
            ))}
          </div>
        )}

        {!loading && projects.length === 0 && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-12 text-center">
            <div className="flex flex-col items-center gap-3">
              <span className="text-5xl">📦</span>
              <p className="text-base font-medium text-gray-600">No projects yet.</p>
              <p className="text-sm text-gray-400">
                Add your first project to get started.
              </p>
              <Link
                to="/projects/new"
                className="mt-2 inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
              >
                + Add Project
              </Link>
            </div>
          </div>
        )}

        {!loading && projects.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((p) => {
              const isIndexing = p.status === 'cloning' || p.status === 'indexing'
              const canReindex = p.status === 'indexed' || p.status === 'failed'
              return (
                <div
                  key={p.id}
                  className="bg-white rounded-2xl shadow-sm border border-gray-200 p-5 flex flex-col"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <Link
                      to={`/projects/${p.id}`}
                      className="text-base font-semibold text-gray-900 hover:text-blue-700 truncate"
                      title={p.repo_name}
                    >
                      {p.repo_name}
                    </Link>
                    <StatusBadge status={p.status} />
                  </div>

                  <a
                    href={p.repo_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:underline truncate mb-3 font-mono"
                  >
                    {p.repo_url}
                  </a>

                  {isIndexing && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                        <span>{p.status === 'cloning' ? 'Cloning…' : 'Indexing…'}</span>
                        <span>{Math.round(p.progress_pct)}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(0, p.progress_pct))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {p.status === 'indexed' && (
                    <div className="text-xs text-gray-600 mb-3">
                      {p.file_count.toLocaleString()} files,{' '}
                      {p.node_count.toLocaleString()} nodes
                    </div>
                  )}

                  <div className="text-xs text-gray-400 mb-4">
                    Last indexed: {formatDate(p.last_indexed_at)}
                  </div>

                  <div className="mt-auto flex items-center gap-2">
                    <Link
                      to={`/projects/${p.id}`}
                      className="px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-medium rounded-md transition-colors"
                    >
                      View →
                    </Link>
                    {canReindex && (
                      <button
                        onClick={() => handleReindex(p.id)}
                        className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-medium rounded-md transition-colors"
                      >
                        Re-index
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(p.id, p.repo_name)}
                      className="ml-auto px-3 py-1.5 text-red-600 hover:bg-red-50 text-xs font-medium rounded-md transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </main>
    </div>
  )
}
