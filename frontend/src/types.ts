export interface FileReviewResult {
  file_path: string
  phase1_found: boolean
  phase2_fault: string
  patch: string
  applies_cleanly: boolean
  reflexion_attempts: number
  risk_level: string // "low" | "medium" | "high" | "unknown"
}

export interface ReviewRecord {
  id: string
  pr_url: string
  repo_name: string
  pr_number: number
  status: string // "pending" | "running" | "completed" | "failed"
  created_at: string
  updated_at: string
  file_reviews: FileReviewResult[]
  total_patches: number
  error?: string
}

export interface ReviewSummary {
  id: string
  pr_url: string
  repo_name: string
  pr_number: number
  status: string
  created_at: string
  total_patches: number
  total_files: number
}

export interface SSEEvent {
  type: string // "status" | "file_start" | "phase" | "done" | "error" | "warning"
  [key: string]: unknown
}

export interface ProjectSummary {
  id: string
  repo_name: string
  repo_url: string
  status: string // "pending" | "cloning" | "indexing" | "indexed" | "failed"
  progress_pct: number
  node_count: number
  file_count: number
  last_indexed_at: string | null
  created_at: string
}

export interface ChatSessionSummary {
  id: string
  title: string
  user_login: string
  message_count: number
  last_message_preview: string
  created_at: string
  updated_at: string
}

export interface ChatMessageRecord {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  tool_name?: string | null
  render_data?: Record<string, unknown> | null
  created_at: string
}

export interface ChatSessionFull {
  id: string
  title: string
  user_login: string
  created_at: string
  updated_at: string
  messages: ChatMessageRecord[]
}

export interface ProjectRecord {
  id: string
  repo_name: string
  repo_url: string
  default_branch: string
  status: string
  progress_pct: number
  error: string | null
  node_count: number
  edge_count: number
  file_count: number
  last_commit_sha: string | null
  last_indexed_at: string | null
  created_at: string
  updated_at: string
}
