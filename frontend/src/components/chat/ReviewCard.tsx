import { Link } from 'react-router-dom'
import StatusBadge from '../StatusBadge'

interface ReviewCardProps {
  reviewId: string
  repoName: string
  prNumber: number
  prUrl?: string
  status?: string
  totalPatches?: number
  totalFiles?: number
  compact?: boolean
}

export default function ReviewCard({
  reviewId,
  repoName,
  prNumber,
  status,
  totalPatches,
  totalFiles,
  compact,
}: ReviewCardProps) {
  return (
    <Link
      to={`/review/${reviewId}`}
      className={`block bg-white border border-gray-200 rounded-2xl shadow-sm hover:shadow-md hover:border-blue-300 transition-all ${
        compact ? 'px-4 py-3' : 'px-4 py-4'
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-base">🔍</span>
            <span className="font-semibold text-gray-900 truncate">
              {repoName} <span className="text-gray-500 font-normal">#{prNumber}</span>
            </span>
            {status && <StatusBadge status={status} />}
          </div>
          {(totalFiles !== undefined || totalPatches !== undefined) && (
            <div className="mt-1 text-xs text-gray-500 flex items-center gap-3">
              {totalFiles !== undefined && <span>{totalFiles} files</span>}
              {totalFiles !== undefined && totalPatches !== undefined && <span>·</span>}
              {totalPatches !== undefined && <span>{totalPatches} patches</span>}
            </div>
          )}
        </div>
        <span className="text-blue-600 text-sm">→</span>
      </div>
    </Link>
  )
}
