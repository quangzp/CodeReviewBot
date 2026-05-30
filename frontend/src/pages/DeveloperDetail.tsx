import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, DeveloperProfileData } from '../api/client'

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  } catch {
    return iso
  }
}

function confidenceColor(c: number): string {
  if (c >= 0.7) return 'bg-red-100 text-red-700 border-red-200'
  if (c >= 0.4) return 'bg-yellow-100 text-yellow-700 border-yellow-200'
  return 'bg-gray-100 text-gray-700 border-gray-200'
}

export default function DeveloperDetail() {
  const { login } = useParams<{ login: string }>()
  const [data, setData] = useState<DeveloperProfileData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!login) return
    api.getDeveloper(login)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [login])

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-gray-400">Loading...</div>
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow p-8 max-w-md text-center">
          <div className="text-5xl mb-3">😞</div>
          <h2 className="font-semibold text-gray-900 mb-2">No profile yet</h2>
          <p className="text-sm text-gray-500 mb-4">{error || 'This developer has no memory profile.'}</p>
          <Link to="/developers" className="text-blue-600 hover:underline text-sm">
            ← Back to developers
          </Link>
        </div>
      </div>
    )
  }

  const { profile, recent_reviews } = data
  const { developer, patterns } = profile
  const activePatterns = patterns.filter(p => p.is_active)

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4">
          <Link to="/developers" className="text-sm text-gray-500 hover:text-gray-900">
            ← All developers
          </Link>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        {/* Profile header */}
        <div className="bg-white rounded-2xl shadow-sm p-6 flex items-center gap-4">
          {developer.avatar_url ? (
            <img src={developer.avatar_url} alt={developer.login} className="w-20 h-20 rounded-full" />
          ) : (
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 text-white flex items-center justify-center text-2xl font-bold">
              {developer.login.slice(0, 2).toUpperCase()}
            </div>
          )}
          <div>
            <h1 className="text-2xl font-bold text-gray-900">@{developer.login}</h1>
            {developer.name && <p className="text-gray-500">{developer.name}</p>}
            <div className="flex gap-4 mt-2 text-sm text-gray-600">
              <span><b>{developer.pr_count}</b> PRs reviewed</span>
              <span>First seen: {formatDate(developer.first_seen_at)}</span>
              <span>Last seen: {formatDate(developer.last_seen_at)}</span>
            </div>
          </div>
        </div>

        {/* Learned patterns */}
        <section className="bg-white rounded-2xl shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900">🧠 Learned Patterns</h2>
            <span className="text-xs text-gray-400">
              Bug patterns the bot has observed this developer making
            </span>
          </div>

          {activePatterns.length === 0 ? (
            <div className="text-center py-8 text-gray-400 text-sm">
              No patterns observed yet. The bot needs at least 3 reviews with the same pattern to learn it.
            </div>
          ) : (
            <ul className="space-y-3">
              {activePatterns.map((p) => (
                <li key={p.pattern.id} className="border border-gray-100 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold text-gray-900">{p.pattern.name}</div>
                      <div className="text-xs text-gray-500 font-mono">{p.pattern.id}</div>
                    </div>
                    <span className={`text-xs px-3 py-1 rounded-full border ${confidenceColor(p.confidence)}`}>
                      {(p.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                  <div className="flex gap-4 mt-2 text-xs text-gray-500">
                    <span>{p.evidence_count} observation{p.evidence_count !== 1 ? 's' : ''}</span>
                    <span>last seen: {formatDate(p.last_observed_at)}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Recent reviews */}
        <section className="bg-white rounded-2xl shadow-sm p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">📜 Recent Reviews</h2>
          {recent_reviews.length === 0 ? (
            <div className="text-center py-6 text-gray-400 text-sm">No reviews yet.</div>
          ) : (
            <ul className="divide-y divide-gray-100">
              {recent_reviews.map((r) => (
                <li key={r.id} className="py-3 flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <Link to={`/review/${r.id}`} className="text-sm font-medium text-gray-900 hover:text-blue-600">
                      {r.repo_name} #{r.pr_number}
                    </Link>
                    <p className="text-xs text-gray-500 mt-1 truncate">{r.summary}</p>
                    {r.patterns.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {r.patterns.map(p => (
                          <span key={p} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded font-mono">
                            {p}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-gray-400 whitespace-nowrap">
                    {formatDate(r.reviewed_at)}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </div>
  )
}
