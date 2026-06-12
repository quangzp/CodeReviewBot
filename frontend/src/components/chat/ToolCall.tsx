import { useState } from 'react'

interface ToolCallProps {
  name: string
  args: Record<string, unknown>
}

export default function ToolCall({ name, args }: ToolCallProps) {
  const [open, setOpen] = useState(false)
  const hasArgs = args && Object.keys(args).length > 0
  return (
    <div className="text-xs text-gray-500 my-2">
      <button
        onClick={() => hasArgs && setOpen((v) => !v)}
        className={`inline-flex items-center gap-1.5 ${hasArgs ? 'hover:text-gray-700 cursor-pointer' : 'cursor-default'}`}
      >
        <span>🛠</span>
        <span>
          calling <code className="font-mono font-semibold text-gray-700">{name}</code>...
        </span>
        {hasArgs && <span className="text-gray-400">{open ? '▾' : '▸'}</span>}
      </button>
      {open && hasArgs && (
        <pre className="mt-1 ml-5 p-2 bg-gray-50 border border-gray-200 rounded text-xs font-mono text-gray-700 overflow-x-auto">
          {JSON.stringify(args, null, 2)}
        </pre>
      )}
    </div>
  )
}
