import { useState } from 'react'
import PatchViewer from '../PatchViewer'

interface FileResult {
  file_path: string
  fault_description?: string
  patch?: string
  applies_cleanly?: boolean
  eval_score?: number
  eval_reason?: string
  risk_level?: string
  attempts?: number
  review_comments?: { severity: string; line?: number; message: string }[]
}

interface ReviewDetailCardProps {
  reviewId?: string
  prUrl?: string
  prNumber?: number
  repoName?: string
  status?: string
  totalPatches?: number
  files?: FileResult[]
}

const RISK_STYLES: Record<string, string> = {
  high:    'bg-red-100 text-red-700 border-red-300',
  medium:  'bg-yellow-100 text-yellow-700 border-yellow-300',
  low:     'bg-green-100 text-green-700 border-green-300',
  unknown: 'bg-gray-100 text-gray-600 border-gray-300',
}

const SEV_DOT: Record<string, string> = {
  bug:        'bg-red-500',
  warning:    'bg-yellow-500',
  suggestion: 'bg-blue-400',
}

function FileRow({ file }: { file: FileResult }) {
  const [expanded, setExpanded] = useState(!!file.patch)
  const [toast, setToast] = useState<string | null>(null)
  const hasPatch = !!file.patch

  const copyPatch = async () => {
    if (!file.patch) return
    try {
      await navigator.clipboard.writeText(file.patch)
      setToast('Copied!')
      setTimeout(() => setToast(null), 2000)
    } catch {
      setToast('Failed')
      setTimeout(() => setToast(null), 2000)
    }
  }

  const savePatch = () => {
    if (!file.patch) return
    const blob = new Blob([file.patch], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file.file_path.replace(/[/\\]/g, '_')}.patch`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      {/* File header row */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full px-4 py-2.5 bg-gray-50 hover:bg-gray-100 flex items-center gap-3 text-left transition-colors"
      >
        <span className="text-sm">{hasPatch ? '✅' : '⚪'}</span>
        <code className="flex-1 text-xs font-mono font-semibold text-gray-800 truncate">
          {file.file_path}
        </code>
        <div className="flex items-center gap-2 shrink-0">
          {file.risk_level && file.risk_level !== 'unknown' && (
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${RISK_STYLES[file.risk_level] ?? RISK_STYLES.unknown}`}
            >
              {file.risk_level}
            </span>
          )}
          {file.eval_score !== undefined && file.eval_score !== null && (
            <span className="text-xs text-gray-500 font-medium">
              score {file.eval_score}/5
            </span>
          )}
          {hasPatch && (
            <span className="text-xs font-medium text-green-600 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full">
              patch ready
            </span>
          )}
          <span className="text-gray-400 text-xs">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 py-3 space-y-3 text-sm">
          {/* Fault description */}
          {file.fault_description && (
            <div>
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Fault identified</div>
              <p className="text-gray-700 text-sm leading-relaxed">{file.fault_description}</p>
            </div>
          )}

          {/* Meta row */}
          {(file.attempts !== undefined || file.applies_cleanly !== undefined) && (
            <div className="flex items-center gap-3 text-xs text-gray-500">
              {file.attempts !== undefined && (
                <span>Attempts: <strong className="text-gray-700">{file.attempts}</strong></span>
              )}
              {file.applies_cleanly !== undefined && (
                <span className={file.applies_cleanly ? 'text-green-600' : 'text-red-600'}>
                  {file.applies_cleanly ? '✓ applies cleanly' : '✗ does not apply cleanly'}
                </span>
              )}
            </div>
          )}

          {/* Review comments */}
          {file.review_comments && file.review_comments.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                Review comments ({file.review_comments.length})
              </div>
              <div className="space-y-1">
                {file.review_comments.map((c, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span
                      className={`mt-1.5 w-2 h-2 rounded-full shrink-0 ${SEV_DOT[c.severity] ?? SEV_DOT.suggestion}`}
                    />
                    <p className="text-xs text-gray-700 leading-relaxed">
                      {c.line ? <span className="text-gray-400 font-mono mr-1">L{c.line}</span> : null}
                      {c.message}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patch viewer */}
          {file.patch && file.file_path && (
            <div>
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Patch</div>
              <PatchViewer patch={file.patch} filePath={file.file_path} />
            </div>
          )}

          {/* Patch actions */}
          {file.patch && (
            <div className="flex items-center gap-2 flex-wrap pt-1">
              <button
                onClick={copyPatch}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 text-xs font-medium rounded-md transition-colors"
              >
                📋 Copy patch
              </button>
              <button
                onClick={savePatch}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 text-xs font-medium rounded-md transition-colors"
              >
                📂 Save to file
              </button>
              {toast && <span className="text-xs text-gray-500 animate-pulse">{toast}</span>}
            </div>
          )}

          {!file.patch && !file.fault_description && (
            <p className="text-xs text-gray-400 italic">No fault identified for this file.</p>
          )}
        </div>
      )}
    </div>
  )
}

export default function ReviewDetailCard({
  prUrl,
  prNumber,
  repoName,
  status,
  totalPatches,
  files = [],
}: ReviewDetailCardProps) {
  const statusLabel = status === 'completed' ? '✅ Completed' : status === 'failed' ? '❌ Failed' : status ?? ''

  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-base">🔍</span>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-gray-900">
              {repoName ?? 'Review'} {prNumber !== undefined ? `#${prNumber}` : ''}
            </div>
            {prUrl && (
              <a
                href={prUrl}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-blue-500 hover:underline truncate block"
              >
                {prUrl}
              </a>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <span>{statusLabel}</span>
          <span className="text-gray-300">·</span>
          <span>
            <strong className="text-gray-800">{totalPatches ?? 0}</strong>/{files.length} files patched
          </span>
        </div>
      </div>

      {/* File list */}
      <div className="p-3 space-y-2">
        {files.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-4">No file results available.</p>
        ) : (
          files.map((f, i) => <FileRow key={i} file={f} />)
        )}
      </div>
    </div>
  )
}
