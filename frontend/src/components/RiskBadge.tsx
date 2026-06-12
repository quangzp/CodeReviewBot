interface RiskBadgeProps {
  level: string
}

const RISK_STYLES: Record<string, string> = {
  low: 'bg-green-100 text-green-700 border-green-300',
  medium: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  high: 'bg-red-100 text-red-700 border-red-300',
  unknown: 'bg-gray-100 text-gray-600 border-gray-300',
}

const RISK_ICONS: Record<string, string> = {
  low: '✓',
  medium: '⚠',
  high: '✗',
  unknown: '?',
}

export default function RiskBadge({ level }: RiskBadgeProps) {
  const normalized = level.toLowerCase()
  const style = RISK_STYLES[normalized] ?? RISK_STYLES.unknown
  const icon = RISK_ICONS[normalized] ?? RISK_ICONS.unknown

  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${style}`}
    >
      <span>{icon}</span>
      {normalized.charAt(0).toUpperCase() + normalized.slice(1)} Risk
    </span>
  )
}
