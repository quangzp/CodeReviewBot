import type { ToolRender } from '../../api/client'
import ProjectCard from './ProjectCard'
import ProjectListCard from './ProjectListCard'
import ReviewCard from './ReviewCard'
import ReviewListCard from './ReviewListCard'
import FixAttemptCard from './FixAttemptCard'

interface ToolResultProps {
  summary: string
  render: ToolRender
}

function AutoOnboardCard({ project, message }: { project?: any; message: string }) {
  return (
    <div className="space-y-2">
      {project && (
        <ProjectCard project={project} indexing={true} />
      )}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl px-4 py-3 flex items-start gap-2">
        <span className="text-base">ℹ️</span>
        <p className="text-sm text-blue-800">{message}</p>
      </div>
    </div>
  )
}

function ErrorCard({ message, suggestion }: { message: string; suggestion?: string }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-2xl px-4 py-3">
      <div className="flex items-start gap-2">
        <span className="text-base">⚠️</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm text-red-800 font-medium">{message}</p>
          {suggestion && <p className="text-xs text-red-700 mt-1">{suggestion}</p>}
        </div>
      </div>
    </div>
  )
}

export default function ToolResult({ summary, render }: ToolResultProps) {
  let body: JSX.Element | null = null

  switch (render.kind) {
    case 'project':
      body = (
        <ProjectCard
          project={render.project}
          alreadyExists={render.already_exists}
          indexing={render.indexing}
        />
      )
      break
    case 'project_list':
      body = <ProjectListCard projects={render.projects} />
      break
    case 'review':
      body = (
        <ReviewCard
          reviewId={render.review_id}
          repoName={render.repo_name}
          prNumber={render.pr_number}
          prUrl={render.pr_url}
        />
      )
      break
    case 'review_list':
      body = <ReviewListCard reviews={render.reviews} />
      break
    case 'fix_attempt':
      body = (
        <FixAttemptCard
          status={render.status}
          filePath={render.file_path}
          faultDescription={render.fault_description}
          patch={render.patch}
          attempts={render.attempts}
          appliesCleanly={render.applies_cleanly}
          repoName={render.repo_name}
          lastError={render.last_error}
          bugDescription={render.bug_description}
        />
      )
      break
    case 'auto_onboard':
      body = <AutoOnboardCard project={render.project} message={render.message} />
      break
    case 'error':
      body = <ErrorCard message={render.message} suggestion={render.suggestion} />
      break
  }

  return (
    <div className="my-2 space-y-1">
      {summary && <div className="text-xs text-gray-500 px-1">{summary}</div>}
      {body}
    </div>
  )
}
