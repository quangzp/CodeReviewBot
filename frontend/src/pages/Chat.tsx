import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, streamChat, type ChatMessage, type ToolRender } from '../api/client'
import type { ChatMessageRecord, ChatSessionSummary } from '../types'
import { useAuth } from '../context/AuthContext'
import MessageBubble from '../components/chat/MessageBubble'
import ToolCall from '../components/chat/ToolCall'
import ToolResult from '../components/chat/ToolResult'
import ChatInput from '../components/chat/ChatInput'

interface ChatItem {
  id: string
  role: 'user' | 'assistant'
  parts: AssistantPart[]
  text?: string
}

type AssistantPart =
  | { kind: 'text'; text: string }
  | { kind: 'tool_call'; name: string; args: Record<string, unknown> }
  | { kind: 'tool_result'; summary: string; render: ToolRender }
  | { kind: 'thinking'; step: number }

const WELCOME: ChatItem = {
  id: 'welcome',
  role: 'assistant',
  parts: [
    {
      kind: 'text',
      text:
        "Hi! I'm your code review assistant. I can:\n\n• Index a repository (give me a GitHub URL)\n• List your projects and reviews\n• Run a review on a pull request\n• Diagnose and propose patches for bugs\n\nWhat would you like to do?",
    },
  ],
}

let idCounter = 0
const nextId = () => `m${Date.now()}_${idCounter++}`

// ---------------------------------------------------------------------------
// Time grouping for sidebar sessions
// ---------------------------------------------------------------------------
type SessionGroup = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Previous 30 days' | 'Older'
const GROUP_ORDER: SessionGroup[] = ['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older']

function groupSessionsByDate(sessions: ChatSessionSummary[]): Record<SessionGroup, ChatSessionSummary[]> {
  const groups: Record<SessionGroup, ChatSessionSummary[]> = {
    Today: [],
    Yesterday: [],
    'Previous 7 days': [],
    'Previous 30 days': [],
    Older: [],
  }
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const dayMs = 24 * 60 * 60 * 1000

  for (const s of sessions) {
    const t = new Date(s.updated_at).getTime()
    if (t >= startOfToday) groups.Today.push(s)
    else if (t >= startOfToday - dayMs) groups.Yesterday.push(s)
    else if (t >= startOfToday - 7 * dayMs) groups['Previous 7 days'].push(s)
    else if (t >= startOfToday - 30 * dayMs) groups['Previous 30 days'].push(s)
    else groups.Older.push(s)
  }
  return groups
}

// ---------------------------------------------------------------------------
// Convert stored ChatMessageRecord[] -> in-memory ChatItem[]
// ---------------------------------------------------------------------------
function recordsToItems(records: ChatMessageRecord[]): ChatItem[] {
  return records.map((r) => {
    if (r.role === 'user') {
      return { id: r.id, role: 'user', parts: [], text: r.content }
    }
    if (r.tool_name && r.render_data) {
      const render = r.render_data as unknown as ToolRender
      return {
        id: r.id,
        role: 'assistant',
        parts: [{ kind: 'tool_result', summary: r.content, render }],
      }
    }
    return {
      id: r.id,
      role: 'assistant',
      parts: [{ kind: 'text', text: r.content }],
    }
  })
}

export default function Chat() {
  const [items, setItems] = useState<ChatItem[]>([WELCOME])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const cancelRef = useRef<(() => void) | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const { user, authEnabled, logout } = useAuth()

  // Auto-scroll on new content
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [items, streaming])

  // Cancel on unmount
  useEffect(() => {
    return () => {
      if (cancelRef.current) cancelRef.current()
    }
  }, [])

  const refreshSessions = async (): Promise<ChatSessionSummary[]> => {
    try {
      const list = await api.listSessions()
      setSessions(list)
      return list
    } catch (e) {
      console.error('Failed to load sessions', e)
      return []
    }
  }

  // Initial load: fetch sessions; create one if empty
  useEffect(() => {
    ;(async () => {
      const list = await refreshSessions()
      if (list.length === 0) {
        try {
          const created = await api.createSession()
          setActiveSessionId(created.id)
          setItems([WELCOME])
          await refreshSessions()
        } catch (e) {
          console.error('Failed to create initial session', e)
        }
      } else {
        // Load most-recently-updated session
        const newest = list[0]
        setActiveSessionId(newest.id)
        try {
          const full = await api.getSession(newest.id)
          const loaded = recordsToItems(full.messages)
          setItems(loaded.length > 0 ? loaded : [WELCOME])
        } catch (e) {
          console.error('Failed to load session', e)
          setItems([WELCOME])
        }
      }
    })()
  }, [])

  const handleSelectSession = async (id: string) => {
    if (id === activeSessionId || streaming) return
    setError(null)
    setActiveSessionId(id)
    setSidebarOpen(false)
    try {
      const full = await api.getSession(id)
      const loaded = recordsToItems(full.messages)
      setItems(loaded.length > 0 ? loaded : [WELCOME])
    } catch (e) {
      console.error('Failed to load session', e)
      setItems([WELCOME])
    }
  }

  const handleNewChat = async () => {
    if (streaming) return
    setError(null)
    try {
      const created = await api.createSession()
      setActiveSessionId(created.id)
      setItems([WELCOME])
      await refreshSessions()
      setSidebarOpen(false)
    } catch (e) {
      console.error('Failed to create session', e)
    }
  }

  const handleRenameSession = async (id: string, currentTitle: string) => {
    const next = window.prompt('Rename chat', currentTitle)
    if (!next || next.trim() === '' || next === currentTitle) return
    try {
      await api.renameSession(id, next.trim())
      await refreshSessions()
    } catch (e) {
      console.error('Failed to rename', e)
    }
  }

  const handleDeleteSession = async (id: string) => {
    if (!window.confirm('Delete this chat? This cannot be undone.')) return
    try {
      await api.deleteSession(id)
      const remaining = await refreshSessions()
      if (id === activeSessionId) {
        if (remaining.length > 0) {
          await handleSelectSession(remaining[0].id)
        } else {
          const created = await api.createSession()
          setActiveSessionId(created.id)
          setItems([WELCOME])
          await refreshSessions()
        }
      }
    } catch (e) {
      console.error('Failed to delete', e)
    }
  }

  const appendToAssistant = (assistantId: string, update: (parts: AssistantPart[]) => AssistantPart[]) => {
    setItems((prev) =>
      prev.map((it) => (it.id === assistantId ? { ...it, parts: update(it.parts) } : it)),
    )
  }

  const handleSend = (text: string) => {
    if (streaming) return
    setError(null)

    const history: ChatMessage[] = items
      .filter((it) => it.id !== 'welcome')
      .map((it) => {
        if (it.role === 'user') {
          return { role: 'user' as const, content: it.text ?? '' }
        }
        const textParts = it.parts.filter((p): p is { kind: 'text'; text: string } => p.kind === 'text')
        return { role: 'assistant' as const, content: textParts.map((p) => p.text).join('') }
      })

    const userId = nextId()
    const assistantId = nextId()
    setItems((prev) => [
      ...prev,
      { id: userId, role: 'user', parts: [], text },
      { id: assistantId, role: 'assistant', parts: [] },
    ])
    setStreaming(true)

    const handle = streamChat(
      text,
      history,
      (ev) => {
        if (ev.type === 'session') {
          setActiveSessionId(ev.id)
          refreshSessions()
        } else if (ev.type === 'message') {
          appendToAssistant(assistantId, (parts) => {
            const last = parts[parts.length - 1]
            if (last && last.kind === 'text') {
              return [...parts.slice(0, -1), { kind: 'text', text: last.text + ev.text }]
            }
            return [...parts, { kind: 'text', text: ev.text }]
          })
        } else if (ev.type === 'tool_call') {
          appendToAssistant(assistantId, (parts) => [
            ...parts,
            { kind: 'tool_call', name: ev.name, args: ev.args },
          ])
        } else if (ev.type === 'tool_result') {
          appendToAssistant(assistantId, (parts) => [
            ...parts,
            { kind: 'tool_result', summary: ev.summary, render: ev.render },
          ])
        } else if (ev.type === 'thinking') {
          appendToAssistant(assistantId, (parts) => [...parts, { kind: 'thinking', step: ev.step }])
        }
      },
      () => {
        setStreaming(false)
        cancelRef.current = null
        refreshSessions()
      },
      (err) => {
        setStreaming(false)
        cancelRef.current = null
        setError(err)
      },
      activeSessionId ?? undefined,
    )
    cancelRef.current = handle.cancel
  }

  const grouped = groupSessionsByDate(sessions)

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/30 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          md:translate-x-0
          fixed md:relative z-40 md:z-auto
          w-[260px] h-full flex-shrink-0
          bg-gray-50 border-r border-gray-200
          flex flex-col
          transition-transform duration-200
        `}
      >
        {/* Header */}
        <div className="px-3 py-3 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">🤖</span>
            <h1 className="text-sm font-bold text-gray-900 flex-1 truncate">GraphRAG Chat</h1>
          </div>
          <div className="flex items-center gap-1">
            <Link
              to="/reviews"
              title="Reviews"
              className="flex-1 text-center px-2 py-1.5 bg-white hover:bg-gray-100 border border-gray-200 text-gray-700 text-xs font-medium rounded-md transition-colors"
            >
              🔍
            </Link>
            <Link
              to="/projects"
              title="Projects"
              className="flex-1 text-center px-2 py-1.5 bg-white hover:bg-gray-100 border border-gray-200 text-gray-700 text-xs font-medium rounded-md transition-colors"
            >
              🗂️
            </Link>
            <Link
              to="/developers"
              title="Developers"
              className="flex-1 text-center px-2 py-1.5 bg-white hover:bg-gray-100 border border-gray-200 text-gray-700 text-xs font-medium rounded-md transition-colors"
            >
              🧠
            </Link>
          </div>
        </div>

        {/* New chat button */}
        <div className="px-3 pt-3 pb-2">
          <button
            onClick={handleNewChat}
            disabled={streaming}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <span className="text-lg leading-none">+</span>
            <span>New chat</span>
          </button>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto px-2 py-2">
          {sessions.length === 0 && (
            <div className="px-2 py-4 text-xs text-gray-400 text-center">No chats yet</div>
          )}
          {GROUP_ORDER.map((group) => {
            const items = grouped[group]
            if (items.length === 0) return null
            return (
              <div key={group} className="mb-3">
                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  {group}
                </div>
                <ul className="space-y-0.5">
                  {items.map((s) => {
                    const isActive = s.id === activeSessionId
                    return (
                      <li key={s.id}>
                        <div
                          onClick={() => handleSelectSession(s.id)}
                          className={`
                            group relative cursor-pointer rounded-md px-2 py-1.5
                            ${
                              isActive
                                ? 'bg-white shadow-sm border-l-4 border-blue-500 pl-2'
                                : 'hover:bg-gray-100 border-l-4 border-transparent'
                            }
                          `}
                        >
                          <div className="min-w-0 pr-12">
                            <div className="text-sm font-medium text-gray-800 truncate">
                              {s.title || 'Untitled'}
                            </div>
                            {s.last_message_preview && (
                              <div className="text-xs text-gray-500 truncate">
                                {s.last_message_preview}
                              </div>
                            )}
                          </div>
                          <div className="absolute right-1 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-1">
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleRenameSession(s.id, s.title)
                              }}
                              title="Rename"
                              className="p-1 rounded hover:bg-gray-200 text-gray-500 hover:text-gray-800"
                            >
                              ✏️
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDeleteSession(s.id)
                              }}
                              title="Delete"
                              className="p-1 rounded hover:bg-gray-200 text-gray-500 hover:text-red-600"
                            >
                              🗑️
                            </button>
                          </div>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </div>
            )
          })}
        </div>

        {/* User footer */}
        {authEnabled && user && (
          <div className="border-t border-gray-200 px-3 py-2 flex items-center gap-2">
            {user.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={user.login}
                className="w-7 h-7 rounded-full border border-gray-200 flex-shrink-0"
              />
            ) : (
              <div className="w-7 h-7 rounded-full bg-gray-300 flex-shrink-0" />
            )}
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium text-gray-800 truncate">{user.login}</div>
            </div>
            <button
              onClick={logout}
              className="text-[10px] text-gray-400 hover:text-red-500 transition-colors px-1 py-1"
              title="Sign out"
            >
              Sign out
            </button>
          </div>
        )}
      </aside>

      {/* Right pane */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar */}
        <div className="md:hidden bg-white border-b border-gray-200 px-3 py-2 flex items-center gap-2">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded hover:bg-gray-100"
            aria-label="Open sidebar"
          >
            ☰
          </button>
          <span className="text-sm font-medium text-gray-800">GraphRAG Chat</span>
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 md:px-6 py-6 space-y-5">
            {items.map((item) => {
              if (item.role === 'user') {
                return <MessageBubble key={item.id} role="user" content={item.text ?? ''} />
              }
              const hasAnyContent = item.parts.length > 0
              return (
                <div key={item.id} className="space-y-2">
                  {!hasAnyContent && streaming && item.id !== 'welcome' && (
                    <MessageBubble role="assistant">
                      <div className="flex items-center gap-1 text-gray-400">
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" />
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
                      </div>
                    </MessageBubble>
                  )}
                  {item.parts.map((part, idx) => {
                    if (part.kind === 'text') {
                      return <MessageBubble key={idx} role="assistant" content={part.text} />
                    }
                    if (part.kind === 'tool_call') {
                      return (
                        <div key={idx} className="pl-11">
                          <ToolCall name={part.name} args={part.args} />
                        </div>
                      )
                    }
                    if (part.kind === 'tool_result') {
                      return (
                        <div key={idx} className="pl-11 pr-4">
                          <ToolResult summary={part.summary} render={part.render} />
                        </div>
                      )
                    }
                    if (part.kind === 'thinking') {
                      return (
                        <div key={idx} className="pl-11 text-xs text-gray-400 italic">
                          thinking (step {part.step})...
                        </div>
                      )
                    }
                    return null
                  })}
                </div>
              )
            })}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-2xl px-4 py-3 text-sm text-red-700">
                ⚠️ {error}
              </div>
            )}
          </div>
        </div>

        {/* Sticky input */}
        <div className="bg-white border-t border-gray-200 flex-shrink-0">
          <div className="max-w-4xl mx-auto px-4 md:px-6 py-3">
            <ChatInput disabled={streaming} onSend={handleSend} />
            <div className="text-[10px] text-gray-400 mt-1 pl-1">Cmd+Enter to send</div>
          </div>
        </div>
      </div>
    </div>
  )
}
