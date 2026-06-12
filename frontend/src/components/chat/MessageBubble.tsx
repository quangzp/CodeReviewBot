import { ReactNode } from 'react'
import { useAuth } from '../../context/AuthContext'

interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content?: string
  children?: ReactNode
}

interface Segment {
  type: 'text' | 'code'
  content: string
}

function parseMarkdownLite(text: string): Segment[] {
  const segments: Segment[] = []
  const regex = /```([\s\S]*?)```/g
  let lastIdx = 0
  let m: RegExpExecArray | null
  while ((m = regex.exec(text)) !== null) {
    if (m.index > lastIdx) {
      segments.push({ type: 'text', content: text.slice(lastIdx, m.index) })
    }
    segments.push({ type: 'code', content: m[1].replace(/^\n/, '').replace(/\n$/, '') })
    lastIdx = m.index + m[0].length
  }
  if (lastIdx < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIdx) })
  }
  return segments
}

export default function MessageBubble({ role, content, children }: MessageBubbleProps) {
  const { user } = useAuth()
  const isUser = role === 'user'

  const segments = content ? parseMarkdownLite(content) : []

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center overflow-hidden bg-gray-100 border border-gray-200">
        {isUser ? (
          user?.avatar_url ? (
            <img src={user.avatar_url} alt={user.login} className="w-full h-full object-cover" />
          ) : (
            <span className="text-sm font-medium text-gray-600">
              {user?.login?.charAt(0).toUpperCase() ?? 'U'}
            </span>
          )
        ) : (
          <span className="text-lg">🤖</span>
        )}
      </div>

      <div
        className={`max-w-[90%] md:max-w-[70%] rounded-2xl shadow-sm px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white rounded-tr-sm'
            : 'bg-white border border-gray-200 text-gray-800 rounded-tl-sm'
        }`}
      >
        {content && (
          <div className="text-sm whitespace-pre-wrap break-words leading-relaxed">
            {segments.map((seg, i) =>
              seg.type === 'code' ? (
                <pre
                  key={i}
                  className={`my-2 p-3 rounded-lg overflow-x-auto text-xs font-mono ${
                    isUser ? 'bg-blue-700 text-blue-50' : 'bg-gray-900 text-gray-100'
                  }`}
                >
                  {seg.content}
                </pre>
              ) : (
                <span key={i}>{seg.content}</span>
              ),
            )}
          </div>
        )}
        {children && <div className={content ? 'mt-3' : ''}>{children}</div>}
      </div>
    </div>
  )
}
