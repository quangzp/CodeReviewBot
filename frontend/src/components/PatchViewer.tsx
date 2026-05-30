import { useState } from 'react'

interface PatchViewerProps {
  patch: string
  filePath: string
}

interface DiffLine {
  type: 'added' | 'removed' | 'context' | 'hunk' | 'header'
  content: string
  lineNo?: number
}

function parsePatch(patch: string): DiffLine[] {
  if (!patch) return []
  return patch.split('\n').map((line) => {
    if (line.startsWith('+++') || line.startsWith('---')) {
      return { type: 'header' as const, content: line }
    }
    if (line.startsWith('@@')) {
      return { type: 'hunk' as const, content: line }
    }
    if (line.startsWith('+')) {
      return { type: 'added' as const, content: line.slice(1) }
    }
    if (line.startsWith('-')) {
      return { type: 'removed' as const, content: line.slice(1) }
    }
    return { type: 'context' as const, content: line.startsWith(' ') ? line.slice(1) : line }
  })
}

const LINE_STYLES: Record<DiffLine['type'], string> = {
  added: 'bg-green-50 text-green-900 border-l-4 border-green-400',
  removed: 'bg-red-50 text-red-900 border-l-4 border-red-400',
  hunk: 'bg-blue-50 text-blue-700 font-medium',
  header: 'bg-gray-100 text-gray-500 italic',
  context: 'bg-white text-gray-800',
}

const LINE_PREFIX: Record<DiffLine['type'], string> = {
  added: '+',
  removed: '-',
  hunk: ' ',
  header: ' ',
  context: ' ',
}

export default function PatchViewer({ patch, filePath }: PatchViewerProps) {
  const [expanded, setExpanded] = useState(true)

  if (!patch) {
    return (
      <div className="text-sm text-gray-400 italic px-4 py-2">No patch available.</div>
    )
  }

  const lines = parsePatch(patch)

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2 bg-gray-100 hover:bg-gray-200 transition-colors text-sm font-mono text-gray-700"
      >
        <span className="flex items-center gap-2">
          <span className="text-gray-500">{expanded ? '▾' : '▸'}</span>
          <span className="font-semibold">{filePath}</span>
          <span className="text-xs text-gray-500">
            ({lines.filter((l) => l.type === 'added').length} additions,{' '}
            {lines.filter((l) => l.type === 'removed').length} deletions)
          </span>
        </span>
        <span className="text-xs text-blue-600">{expanded ? 'Collapse' : 'Expand'}</span>
      </button>

      {expanded && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <tbody>
              {lines.map((line, idx) => (
                <tr key={idx} className={LINE_STYLES[line.type]}>
                  <td className="w-6 text-center px-1 select-none text-gray-400 border-r border-gray-200">
                    {LINE_PREFIX[line.type]}
                  </td>
                  <td className="px-3 py-0.5 whitespace-pre-wrap break-all">
                    {line.content || ' '}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
