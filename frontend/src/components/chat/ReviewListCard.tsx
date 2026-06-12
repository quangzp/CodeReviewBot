import type { ReviewSummary } from '../../types'
import ReviewCard from './ReviewCard'

interface ReviewListCardProps {
  reviews: ReviewSummary[]
}

export default function ReviewListCard({ reviews }: ReviewListCardProps) {
  if (reviews.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic px-4 py-3 bg-gray-50 rounded-2xl border border-gray-200">
        No reviews yet.
      </div>
    )
  }
  return (
    <div className="space-y-2">
      {reviews.map((r) => (
        <ReviewCard
          key={r.id}
          reviewId={r.id}
          repoName={r.repo_name}
          prNumber={r.pr_number}
          prUrl={r.pr_url}
          status={r.status}
          totalFiles={r.total_files}
          totalPatches={r.total_patches}
          compact
        />
      ))}
    </div>
  )
}
