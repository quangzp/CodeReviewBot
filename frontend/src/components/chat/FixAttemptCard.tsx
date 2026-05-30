import { useState } from 'react'
import PatchViewer from '../PatchViewer'

interface FixAttemptCardProps {
  status: 'success' | 'no_file' | 'no_content' | 'patch_failed'
  filePath?: string
  faultDescription?: string
  patch?: string
  attempts?: number
  appliesCleanly?: boolean
  repoName?: string
  lastError?: string
  bugDescription?: string
}

const STATUS_LABEL: Record<FixAttemptCardProps['status'], string> = {
  success: 'Patch ready',
  no_file: 'File not found',
  no_content: 'No content',
  patch_failed: 'Patch failed',
}

const STATUS_STYLES: Record<FixAttemptCardProps['status'], string> = {
  success: 'bg-green-100 text-green-700 border-green-300',
  no_file: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  no_content: 'bg-yellow-100 text-yellow-700 border-yellow-300',
  patch_failed: 'bg-red-100 text-red-700 border-red-300',
}

export default function FixAttemptCard({
  status,
  filePath,
  faultDescription,
  patch,
  attempts,
  appliesCleanly,
  repoName,
  lastError,
  bugDescription,
}: FixAttemptCardProps) {
  const [toast, setToast] = useState<string | null>(null)

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2500)
  }

  const copyPatch = async () => {
    if (!patch) return
    try {
      await navigator.clipboard.writeText(patch)
      showToast('Patch copied to clipboard')
    } catch {
      showToast('Failed to copy')
    }
  }

  const saveToFile = () => {
    if (!patch) return
    const blob = new Blob([patch], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const base = filePath ? filePath.replace(/[/\\]/g, '_') : 'fix'
    a.href = url
    a.download = `${base}.patch`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const openPR = () => {
    showToast('Coming soon: open PR with this fix')
  }

  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-base">🔧</span>
          <code className="text-sm font-mono font-semibold text-gray-800 truncate">
            {filePath || (repoName ? repoName : 'fix attempt')}
          </code>
        </div>
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${STATUS_STYLES[status]}`}
        >
          {STATUS_LABEL[status]}
        </span>
      </div>

      <div className="px-4 py-3 space-y-2 text-sm">
        {bugDescription && (
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Bug</span>
            <p className="text-gray-700 mt-0.5">{bugDescription}</p>
          </div>
        )}
        {faultDescription && (
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Fault</span>
            <p className="text-gray-700 mt-0.5">{faultDescription}</p>
          </div>
        )}
        {attempts !== undefined && (
          <div className="text-xs text-gray-500">
            Attempts: <span className="font-semibold text-gray-700">{attempts}</span>
            {appliesCleanly !== undefined && (
              <>
                {' · '}
                {appliesCleanly ? (
                  <span className="text-green-600">applies cleanly</span>
                ) : (
                  <span className="text-red-600">does not apply cleanly</span>
                )}
              </>
            )}
            {attempts > 1 && (
              <span className="text-gray-400 italic"> (Reflexion auto-retried until it applied)</span>
            )}
          </div>
        )}
        {lastError && (
          <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 font-mono whitespace-pre-wrap">
            {lastError}
          </div>
        )}
      </div>

      {patch && filePath && (
        <div className="px-4 pb-3">
          <PatchViewer patch={patch} filePath={filePath} />
        </div>
      )}

      {patch && (
        <div className="px-4 py-3 border-t border-gray-200 bg-gray-50 flex items-center gap-2 flex-wrap">
          <button
            onClick={copyPatch}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 text-xs font-medium rounded-md transition-colors"
          >
            📋 Copy patch
          </button>
          <button
            onClick={saveToFile}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 text-xs font-medium rounded-md transition-colors"
          >
            📂 Save to file
          </button>
          <button
            onClick={openPR}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-md transition-colors"
          >
            🚀 Open PR with this fix
          </button>
          {toast && (
            <span className="text-xs text-gray-500 ml-2 animate-pulse">{toast}</span>
          )}
        </div>
      )}
    </div>
  )
}
