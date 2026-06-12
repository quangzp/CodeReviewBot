import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ReviewSummary } from '../types'
import StatusBadge from '../components/StatusBadge'
import { useAuth } from '../context/AuthContext'

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function SkeletonRow() {
  return (
    <tr className="border-t border-gray-100">
      {[1, 2, 3, 4, 5, 6, 7].map((i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4" />
        </td>
      ))}
    </tr>
  )
}

export default function Dashboard() {
  const [reviews, setReviews] = useState<ReviewSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()
  const { user, authEnabled, logout } = useAuth()

  const fetchReviews = async () => {
    try {
      const data = await api.listReviews()
      setReviews(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reviews')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReviews()
    const interval = setInterval(fetchReviews, 10_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🤖</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900">GraphRAG Code Review Bot</h1>
              <p className="text-xs text-gray-500">Automated PR analysis with knowledge graphs</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              to="/"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors"
            >
              💬 Chat
            </Link>
            <Link
              to="/projects"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors"
            >
              🗂️ Projects
            </Link>
            <Link
              to="/developers"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors"
            >
              🧠 Developers
            </Link>
            <button
              onClick={() => navigate('/new')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg shadow-sm transition-colors"
            >
              <span className="text-base leading-none">+</span>
              New Review
            </button>
            {authEnabled && user && (
              <div className="flex items-center gap-2">
                {user.avatar_url && (
                  <img src={user.avatar_url} alt={user.login} className="w-8 h-8 rounded-full border border-gray-200" />
                )}
                <span className="text-sm text-gray-600 hidden sm:block">{user.login}</span>
                <button
                  onClick={logout}
                  className="text-xs text-gray-400 hover:text-red-500 transition-colors px-2 py-1"
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Reviews</h2>
          <span className="text-xs text-gray-400">Auto-refreshes every 10s</span>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Repo
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  PR #
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Files Reviewed
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Patches
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Created At
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading &&
                [1, 2, 3].map((i) => <SkeletonRow key={i} />)}

              {!loading && reviews.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-16 text-center text-gray-400">
                    <div className="flex flex-col items-center gap-3">
                      <span className="text-4xl">📭</span>
                      <p className="text-base font-medium text-gray-500">No reviews yet.</p>
                      <p className="text-sm">
                        <Link to="/new" className="text-blue-600 hover:underline">
                          Submit a PR
                        </Link>{' '}
                        to get started.
                      </p>
                    </div>
                  </td>
                </tr>
              )}

              {!loading &&
                reviews.map((review) => (
                  <tr
                    key={review.id}
                    className="border-t border-gray-100 hover:bg-gray-50 transition-colors"
                  >
                    <td className="px-4 py-3 font-medium text-gray-900 max-w-xs truncate">
                      {review.repo_name || '—'}
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      #{review.pr_number}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={review.status} />
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      {review.total_files ?? '—'}
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      {review.total_patches ?? '—'}
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">
                      {formatDate(review.created_at)}
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        to={`/review/${review.id}`}
                        className="inline-flex items-center gap-1 px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-medium rounded-md transition-colors"
                      >
                        View →
                      </Link>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  )
}
