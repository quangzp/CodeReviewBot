import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, DeveloperSummary } from '../api/client'

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

export default function Developers() {
  const [devs, setDevs] = useState<DeveloperSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    api.listDevelopers()
      .then(setDevs)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🧠</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Developer Memory</h1>
              <p className="text-xs text-gray-500">What the bot has learned about each contributor</p>
            </div>
          </div>
          <Link
            to="/"
            className="text-sm text-gray-500 hover:text-gray-900"
          >
            ← Back to reviews
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {loading && (
          <div className="text-center text-gray-400 py-12">Loading developers...</div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {!loading && !error && devs.length === 0 && (
          <div className="bg-white rounded-2xl shadow p-12 text-center">
            <div className="text-6xl mb-4">👻</div>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">No developers yet</h2>
            <p className="text-gray-500 text-sm">
              The bot will learn about developers as it reviews their pull requests.
            </p>
          </div>
        )}

        {!loading && devs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {devs.map((dev) => (
              <button
                key={dev.login}
                onClick={() => navigate(`/developer/${dev.login}`)}
                className="bg-white rounded-2xl shadow-sm hover:shadow-md transition-shadow p-6 text-left border border-gray-100 hover:border-blue-300"
              >
                <div className="flex items-center gap-3 mb-3">
                  {dev.avatar_url ? (
                    <img src={dev.avatar_url} alt={dev.login} className="w-12 h-12 rounded-full" />
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 text-white flex items-center justify-center font-bold">
                      {dev.login.slice(0, 2).toUpperCase()}
                    </div>
                  )}
                  <div>
                    <div className="font-semibold text-gray-900">@{dev.login}</div>
                    {dev.name && <div className="text-xs text-gray-500">{dev.name}</div>}
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-gray-100">
                  <span>{dev.pr_count} PR{dev.pr_count !== 1 ? 's' : ''} reviewed</span>
                  <span>last: {formatDate(dev.last_seen_at)}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
