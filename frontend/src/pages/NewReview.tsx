import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ProjectSummary } from '../types'

const PR_URL_REGEX = /^https?:\/\/github\.com\/([^/]+\/[^/]+)\/pull\/\d+\/?$/

function parseRepoFromPrUrl(url: string): string | null {
  const m = url.trim().match(PR_URL_REGEX)
  return m ? m[1] : null
}

export default function NewReview() {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [loadingProjects, setLoadingProjects] = useState(true)
  const [selectedProjectId, setSelectedProjectId] = useState<string>('')
  const [prUrl, setPrUrl] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.listProjects()
      .then(setProjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoadingProjects(false))
  }, [])

  const indexedProjects = useMemo(
    () => projects.filter(p => p.status === 'indexed'),
    [projects]
  )

  const selectedProject = useMemo(
    () => projects.find(p => p.id === selectedProjectId),
    [projects, selectedProjectId]
  )

  // Auto-select project if PR URL matches an onboarded repo
  useEffect(() => {
    const repo = parseRepoFromPrUrl(prUrl)
    if (!repo) return
    const match = projects.find(p => p.repo_name === repo)
    if (match && match.id !== selectedProjectId) {
      setSelectedProjectId(match.id)
    }
  }, [prUrl, projects, selectedProjectId])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setValidationError(null)
    setError(null)

    const trimmed = prUrl.trim()
    if (!trimmed) {
      setValidationError('Please enter a GitHub PR URL.')
      return
    }
    if (!PR_URL_REGEX.test(trimmed)) {
      setValidationError('Invalid URL format. Expected: https://github.com/owner/repo/pull/123')
      return
    }
    if (!selectedProjectId) {
      setValidationError('Please select an indexed project first.')
      return
    }

    // Validate that the PR belongs to the selected project
    const prRepo = parseRepoFromPrUrl(trimmed)
    if (selectedProject && prRepo && prRepo !== selectedProject.repo_name) {
      setValidationError(
        `This PR belongs to ${prRepo}, but you selected ${selectedProject.repo_name}.`
      )
      return
    }

    setSubmitting(true)
    try {
      const record = await api.createReview(trimmed)
      navigate(`/review/${record.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start review')
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-3">
          <Link to="/" className="text-gray-400 hover:text-gray-600 transition-colors text-sm">
            ← Back
          </Link>
          <span className="text-gray-300">|</span>
          <span className="text-2xl">🤖</span>
          <h1 className="text-xl font-bold text-gray-900">New Code Review</h1>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-12">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-1">Start a Review</h2>
          <p className="text-sm text-gray-500 mb-6">
            Pick an indexed project, then paste the PR URL to review.
          </p>

          {/* Project gate */}
          {!loadingProjects && indexedProjects.length === 0 && (
            <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm font-medium text-yellow-900 mb-1">
                ⚠️ No indexed projects yet
              </p>
              <p className="text-xs text-yellow-700 mb-3">
                You need to onboard a project before reviewing its PRs. The bot must build the knowledge graph first.
              </p>
              <Link
                to="/projects/new"
                className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 hover:underline"
              >
                Onboard your first project →
              </Link>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Project selector */}
            <div>
              <label htmlFor="project" className="block text-sm font-medium text-gray-700 mb-1.5">
                Project
              </label>
              <select
                id="project"
                value={selectedProjectId}
                onChange={(e) => setSelectedProjectId(e.target.value)}
                disabled={submitting || loadingProjects || indexedProjects.length === 0}
                className="w-full px-4 py-2.5 rounded-lg border border-gray-300 bg-white text-sm transition-colors outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-400"
              >
                <option value="">
                  {loadingProjects
                    ? 'Loading projects...'
                    : indexedProjects.length === 0
                      ? 'No indexed projects'
                      : 'Select a project...'}
                </option>
                {indexedProjects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.repo_name} ({p.node_count.toLocaleString()} nodes)
                  </option>
                ))}
              </select>
              {selectedProject && (
                <p className="mt-1.5 text-xs text-gray-500">
                  Graph: {selectedProject.file_count.toLocaleString()} files, {selectedProject.node_count.toLocaleString()} nodes
                </p>
              )}
            </div>

            {/* PR URL */}
            <div>
              <label htmlFor="pr-url" className="block text-sm font-medium text-gray-700 mb-1.5">
                GitHub PR URL
              </label>
              <input
                id="pr-url"
                type="url"
                value={prUrl}
                onChange={(e) => { setPrUrl(e.target.value); setValidationError(null) }}
                placeholder={
                  selectedProject
                    ? `https://github.com/${selectedProject.repo_name}/pull/123`
                    : 'https://github.com/owner/repo/pull/123'
                }
                disabled={submitting || indexedProjects.length === 0}
                className={`w-full px-4 py-2.5 rounded-lg border text-sm font-mono transition-colors outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-400 ${validationError ? 'border-red-400 bg-red-50' : 'border-gray-300 bg-white'}`}
              />
              {validationError && (
                <p className="mt-1.5 text-xs text-red-600">{validationError}</p>
              )}
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                <strong>Error:</strong> {error}
              </div>
            )}

            <div className="flex items-center gap-3 pt-2">
              <button
                type="submit"
                disabled={submitting || indexedProjects.length === 0 || !selectedProjectId}
                className="inline-flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
              >
                {submitting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Starting Review...
                  </>
                ) : (
                  'Start Review'
                )}
              </button>
              <Link
                to="/"
                className="px-4 py-2.5 text-sm text-gray-600 hover:text-gray-800 transition-colors"
              >
                Cancel
              </Link>
            </div>
          </form>
        </div>
      </main>
    </div>
  )
}
