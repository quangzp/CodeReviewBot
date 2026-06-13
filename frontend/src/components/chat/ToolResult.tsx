import type { ToolRender } from '../../api/client'
import ProjectCard from './ProjectCard'
import ProjectListCard from './ProjectListCard'
import ReviewCard from './ReviewCard'
import ReviewListCard from './ReviewListCard'
import ReviewDetailCard from './ReviewDetailCard'
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

function ProjectExplorationCard({
  repoName,
  question,
  answer,
  entrypoints,
  tree,
}: {
  repoName: string
  question: string
  answer: string
  entrypoints?: string[]
  tree?: string
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-base">ðŸ§­</span>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-gray-900 truncate">{repoName}</div>
            <div className="text-xs text-gray-500 truncate">{question}</div>
          </div>
        </div>
      </div>

      <div className="px-4 py-3 space-y-3">
        <div className="text-sm text-gray-800 whitespace-pre-wrap leading-6">{answer}</div>

        {entrypoints && entrypoints.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
              Entrypoints
            </div>
            <div className="flex flex-wrap gap-1.5">
              {entrypoints.map((path) => (
                <code
                  key={path}
                  className="px-2 py-1 bg-gray-100 border border-gray-200 rounded text-xs text-gray-700"
                >
                  {path}
                </code>
              ))}
            </div>
          </div>
        )}

        {tree && (
          <details className="group">
            <summary className="cursor-pointer text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Directory tree
            </summary>
            <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-gray-950 text-gray-100 text-xs p-3 leading-5">
              {tree}
            </pre>
          </details>
        )}
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
    case 'project_exploration':
      body = (
        <ProjectExplorationCard
          repoName={render.repo_name}
          question={render.question}
          answer={render.answer}
          entrypoints={render.entrypoints}
          tree={render.tree}
        />
      )
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
    case 'review_detail':
      body = (
        <ReviewDetailCard
          reviewId={render.review_id}
          prUrl={render.pr_url}
          prNumber={render.pr_number}
          repoName={render.repo_name}
          status={render.status}
          totalPatches={render.total_patches}
          files={render.files}
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
