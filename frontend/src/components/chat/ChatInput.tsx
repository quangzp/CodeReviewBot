import { KeyboardEvent, useEffect, useRef, useState } from 'react'

interface ChatInputProps {
  disabled?: boolean
  onSend: (text: string) => void
}

const LINE_HEIGHT = 24 // px, approx for text-sm
const MAX_LINES = 5

export default function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [value, setValue] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const max = LINE_HEIGHT * MAX_LINES + 16
    ta.style.height = `${Math.min(ta.scrollHeight, max)}px`
    ta.style.overflowY = ta.scrollHeight > max ? 'auto' : 'hidden'
  }, [value])

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex items-end gap-2">
      <textarea
        ref={taRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        rows={1}
        placeholder={disabled ? 'Waiting for response...' : 'Ask anything — e.g. "Index foo/bar" or "Fix the null deref in handler.py"'}
        className="flex-1 resize-none px-4 py-3 border border-gray-300 rounded-2xl text-sm leading-6 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="inline-flex items-center gap-1.5 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white text-sm font-medium rounded-2xl shadow-sm transition-colors"
        title="Send (Cmd+Enter)"
      >
        Send
      </button>
    </div>
  )
}
