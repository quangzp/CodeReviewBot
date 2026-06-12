interface StatusBadgeProps {
  status: string
}

const STATUS_STYLES: Record<string, string> = {
  // Review statuses
  pending: 'bg-gray-100 text-gray-700 border-gray-300',
  running: 'bg-blue-100 text-blue-700 border-blue-300',
  completed: 'bg-green-100 text-green-700 border-green-300',
  failed: 'bg-red-100 text-red-700 border-red-300',
  // Project statuses
  cloning: 'bg-blue-100 text-blue-700 border-blue-300',
  indexing: 'bg-blue-100 text-blue-700 border-blue-300',
  indexed: 'bg-green-100 text-green-700 border-green-300',
}

const STATUS_DOTS: Record<string, string> = {
  pending: 'bg-gray-400',
  running: 'bg-blue-500 animate-pulse',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
  cloning: 'bg-blue-500 animate-pulse',
  indexing: 'bg-blue-500 animate-pulse',
  indexed: 'bg-green-500',
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-600 border-gray-200'
  const dot = STATUS_DOTS[status] ?? 'bg-gray-400'

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${style}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}
