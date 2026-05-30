import { useEffect, useRef } from 'react'
import type { SSEEvent } from '../types'

interface PipelineProgressProps {
  events: SSEEvent[]
  connected: boolean
}

interface Step {
  id: string
  icon: string
  label: string
  detail?: string
  state: 'done' | 'running' | 'error'
}

function eventsToSteps(events: SSEEvent[]): Step[] {
  const steps: Step[] = []

  for (const ev of events) {
    switch (ev.type) {
      case 'status': {
        const msg = String(ev.message ?? ev.msg ?? '')
        steps.push({
          id: `status-${steps.length}`,
          icon: 'ℹ',
          label: msg,
          state: 'done',
        })
        break
      }
      case 'file_start': {
        const filePath = String(ev.file_path ?? ev.file ?? '')
        const current = Number(ev.current ?? ev.index ?? 0)
        const total = Number(ev.total ?? 0)
        steps.push({
          id: `file-${filePath}-${steps.length}`,
          icon: '📄',
          label: `Reviewing ${filePath}`,
          detail: total ? `(${current}/${total})` : undefined,
          state: 'running',
        })
        break
      }
      case 'phase': {
        const phase = Number(ev.phase ?? 0)
        const attempt = Number(ev.attempt ?? 0)
        const status = String(ev.status ?? '')
        const fault = String(ev.fault ?? ev.fault_description ?? '')

        if (phase === 1) {
          steps.push({
            id: `phase1-${steps.length}`,
            icon: '🔍',
            label: 'Phase 1: File Localization',
            detail: status,
            state: status === 'failed' ? 'error' : 'done',
          })
        } else if (phase === 2) {
          steps.push({
            id: `phase2-${steps.length}`,
            icon: '🐛',
            label: 'Phase 2: Fault Localization',
            detail: fault || status,
            state: status === 'failed' ? 'error' : 'done',
          })
        } else if (phase === 3) {
          steps.push({
            id: `phase3-${steps.length}-${attempt}`,
            icon: '🔧',
            label: `Phase 3: Patch Generation`,
            detail: attempt ? `attempt ${attempt}/3` : status,
            state: status === 'failed' ? 'error' : 'running',
          })
        }
        break
      }
      case 'done': {
        const patches = Number(ev.patches ?? ev.total_patches ?? 0)
        steps.push({
          id: `done-${steps.length}`,
          icon: '✅',
          label: `Review complete!`,
          detail: patches ? `${patches} patches generated` : undefined,
          state: 'done',
        })
        break
      }
      case 'error': {
        const msg = String(ev.message ?? ev.error ?? ev.msg ?? 'Unknown error')
        steps.push({
          id: `error-${steps.length}`,
          icon: '❌',
          label: `Error: ${msg}`,
          state: 'error',
        })
        break
      }
      case 'warning': {
        const msg = String(ev.message ?? ev.warning ?? '')
        steps.push({
          id: `warning-${steps.length}`,
          icon: '⚠️',
          label: msg,
          state: 'done',
        })
        break
      }
      case 'memory': {
        const msg = String(ev.message ?? '')
        const patterns = (ev.patterns as string[] | undefined) || []
        const detail = patterns.length > 0
          ? `Patterns: ${patterns.join(', ')}`
          : (ev.context ? String(ev.context).slice(0, 100) : undefined)
        steps.push({
          id: `memory-${steps.length}`,
          icon: '🧠',
          label: msg,
          detail,
          state: 'done',
        })
        break
      }
      default:
        break
    }
  }

  return steps
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-blue-500"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

export default function PipelineProgress({ events, connected }: PipelineProgressProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const steps = eventsToSteps(events)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [steps.length])

  if (steps.length === 0 && !connected) {
    return (
      <div className="text-sm text-gray-400 italic py-4 text-center">
        Waiting for pipeline events...
      </div>
    )
  }

  return (
    <div className="relative pl-6">
      {/* Vertical line */}
      <div className="absolute left-3 top-2 bottom-2 w-0.5 bg-gray-200" />

      <div className="space-y-3">
        {steps.map((step, idx) => {
          const isLast = idx === steps.length - 1
          return (
            <div key={step.id} className="relative flex items-start gap-3">
              {/* Node on the timeline */}
              <div
                className={`absolute -left-3 w-3 h-3 rounded-full border-2 mt-0.5 ${
                  step.state === 'error'
                    ? 'bg-red-500 border-red-400'
                    : step.state === 'running'
                    ? 'bg-blue-500 border-blue-400 animate-pulse'
                    : 'bg-green-500 border-green-400'
                }`}
              />

              <div
                className={`ml-2 flex-1 rounded-lg px-3 py-2 text-sm ${
                  step.state === 'error'
                    ? 'bg-red-50 border border-red-200'
                    : step.state === 'running'
                    ? 'bg-blue-50 border border-blue-200'
                    : 'bg-white border border-gray-100'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span>{step.icon}</span>
                  <span
                    className={`font-medium ${
                      step.state === 'error' ? 'text-red-700' : 'text-gray-800'
                    }`}
                  >
                    {step.label}
                  </span>
                  {step.state === 'running' && isLast && connected && <Spinner />}
                </div>
                {step.detail && (
                  <div className="mt-0.5 ml-6 text-xs text-gray-500 truncate max-w-lg">
                    {step.detail}
                  </div>
                )}
              </div>
            </div>
          )
        })}

        {connected && steps.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-blue-600 ml-2">
            <Spinner />
            <span>Connecting to pipeline...</span>
          </div>
        )}
      </div>

      <div ref={bottomRef} />
    </div>
  )
}
