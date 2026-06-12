import { useState, type FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { api } from '../api/client'

const REPO_URL_REGEX = /^https?:\/\/github\.com\/[^/]+\/[^/]+\/?$/

export default function NewProject() {
  const [repoUrl, setRepoUrl] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const navigate = useNavigate()

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setValidationError(null)
    setError(null)

    const trimmed = repoUrl.trim().replace(/\/$/, '')
    if (!trimmed) {
      setValidationError('Please enter a GitHub repository URL.')
      return
    }
    if (!REPO_URL_REGEX.test(trimmed)) {
      setValidationError(
        'Invalid URL format. Expected: https://github.com/owner/repo'
      )
      return
    }

    setSubmitting(true)
    try {
      const record = await api.createProject(trimmed)
      navigate(`/projects/${record.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to onboard project')
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-3">
          <Link to="/projects" className="text-gray-400 hover:text-gray-600 transition-colors text-sm">
            ← Back to projects
          </Link>
          <span className="text-gray-300">|</span>
          <span className="text-2xl">🗂️</span>
          <h1 className="text-xl font-bold text-gray-900">Add Project</h1>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-12">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-1">Onboard a Repository</h2>
          <p className="text-sm text-gray-500 mb-6">
            Provide a GitHub repository URL. We'll clone the repo and index it into the
            knowledge graph so PR reviews can run against it.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 mb-1.5">
                GitHub repo URL
              </label>
              <input
                id="repo-url"
                type="url"
                value={repoUrl}
                onChange={(e) => {
                  setRepoUrl(e.target.value)
                  setValidationError(null)
                }}
                placeholder="https://github.com/owner/repo"
                disabled={submitting}
                className={`w-full px-4 py-2.5 rounded-lg border text-sm font-mono transition-colors outline-none
                  focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                  disabled:bg-gray-50 disabled:text-gray-400
                  ${validationError ? 'border-red-400 bg-red-50' : 'border-gray-300 bg-white'}
                `}
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
                disabled={submitting}
                className="inline-flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
              >
                {submitting ? (
                  <>
                    <svg
                      className="animate-spin h-4 w-4"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    Onboarding...
                  </>
                ) : (
                  'Onboard Project'
                )}
              </button>
              <Link
                to="/projects"
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
