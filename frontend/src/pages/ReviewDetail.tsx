import { useEffect, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, useSSE } from '../api/client'
import type { ReviewRecord } from '../types'
import StatusBadge from '../components/StatusBadge'
import RiskBadge from '../components/RiskBadge'
import PatchViewer from '../components/PatchViewer'
import PipelineProgress from '../components/PipelineProgress'

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function ReviewDetail() {
  const { id } = useParams<{ id: string }>()
  const [review, setReview] = useState<ReviewRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const isActive =
    review?.status === 'pending' || review?.status === 'running'

  const { events, connected } = useSSE(id, isActive)

  const fetchReview = useCallback(async () => {
    if (!id) return
    try {
      const data = await api.getReview(id)
      setReview(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load review')
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    fetchReview()
  }, [fetchReview])

  // Poll when running/pending (fallback if SSE "done" event triggers re-fetch)
  useEffect(() => {
    if (!isActive) return
    const interval = setInterval(fetchReview, 5_000)
    return () => clearInterval(interval)
  }, [isActive, fetchReview])

  // Re-fetch once SSE disconnects (pipeline finished)
  useEffect(() => {
    if (!connected && events.length > 0) {
      fetchReview()
    }
  }, [connected, events.length, fetchReview])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-500">
          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading review...
        </div>
      </div>
    )
  }

  if (error || !review) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 font-medium mb-2">{error ?? 'Review not found'}</p>
          <Link to="/" className="text-blue-600 hover:underline text-sm">← Back to Dashboard</Link>
        </div>
      </div>
    )
  }

  const showPipeline = isActive || events.length > 0
  const showResults =
    (review.status === 'completed' || review.status === 'failed') &&
    review.file_reviews?.length > 0

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-3">
          <Link to="/" className="text-gray-400 hover:text-gray-600 transition-colors text-sm">
            ← Dashboard
          </Link>
          <span className="text-gray-300">|</span>
          <span className="text-2xl">🤖</span>
          <h1 className="text-xl font-bold text-gray-900">Review Detail</h1>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">

        {/* Section A: Review header card */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-3 flex-wrap">
                <StatusBadge status={review.status} />
                <span className="text-sm font-semibold text-gray-700">
                  {review.repo_name}
                </span>
                <span className="text-sm text-gray-500">#{review.pr_number}</span>
              </div>
              <a
                href={review.pr_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 hover:underline text-sm font-mono break-all"
              >
                {review.pr_url} ↗
              </a>
              <p className="text-xs text-gray-400">
                Created {formatDate(review.created_at)}
              </p>
            </div>
            <div className="flex flex-col items-end gap-1 text-sm text-gray-600">
              <div className="flex items-center gap-1.5">
                <span className="text-gray-400">Files:</span>
                <span className="font-semibold">{review.file_reviews?.length ?? 0}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-gray-400">Patches:</span>
                <span className="font-semibold">{review.total_patches ?? 0}</span>
              </div>
            </div>
          </div>

          {review.error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              <strong>Error:</strong> {review.error}
            </div>
          )}
        </div>

        {/* Section B: Live Pipeline Progress */}
        {showPipeline && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center gap-2 mb-4">
              <h2 className="text-base font-semibold text-gray-800">Pipeline Progress</h2>
              {connected && (
                <span className="flex items-center gap-1.5 text-xs text-blue-600 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                  Live
                </span>
              )}
            </div>
            <PipelineProgress events={events} connected={connected} />
          </div>
        )}

        {/* Section C: File Results */}
        {showResults && (
          <div className="space-y-4">
            <h2 className="text-base font-semibold text-gray-800">
              File Reviews ({review.file_reviews.length})
            </h2>

            {review.file_reviews.map((fr, idx) => (
              <div
                key={`${fr.file_path}-${idx}`}
                className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden"
              >
                {/* File header */}
                <div className="px-5 py-4 border-b border-gray-100 flex flex-wrap items-start justify-between gap-3">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-mono font-semibold text-gray-900 break-all">
                        {fr.file_path}
                      </span>
                      <RiskBadge level={fr.risk_level} />
                    </div>
                    {fr.phase2_fault && (
                      <p className="text-sm text-gray-600 mt-1">
                        <span className="font-medium text-gray-500">Fault: </span>
                        {fr.phase2_fault}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1 text-xs text-gray-500 shrink-0">
                    <div className="flex items-center gap-1">
                      {fr.phase1_found ? (
                        <span className="text-green-600">✓ Phase 1: Located</span>
                      ) : (
                        <span className="text-red-500">✗ Phase 1: Not found</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      {fr.applies_cleanly ? (
                        <span className="text-green-600">✓ Patch applies cleanly</span>
                      ) : (
                        <span className="text-yellow-600">⚠ Patch may not apply</span>
                      )}
                    </div>
                    {fr.reflexion_attempts > 0 && (
                      <div className="flex items-center gap-1">
                        <span className="text-blue-600">
                          ✓ Fixed on attempt {fr.reflexion_attempts}/3
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Patch viewer */}
                <div className="p-4">
                  <PatchViewer patch={fr.patch} filePath={fr.file_path} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Completed but no file reviews */}
        {review.status === 'completed' && (!review.file_reviews || review.file_reviews.length === 0) && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center text-gray-400">
            <p className="text-base">No file reviews available for this PR.</p>
          </div>
        )}
      </main>
    </div>
  )
}
