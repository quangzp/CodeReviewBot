import { Link } from 'react-router-dom'
import type { ProjectRecord, ProjectSummary } from '../../types'
import StatusBadge from '../StatusBadge'

interface ProjectCardProps {
  project: ProjectRecord | ProjectSummary
  alreadyExists?: boolean
  indexing?: boolean
  compact?: boolean
}

export default function ProjectCard({ project, alreadyExists, indexing, compact }: ProjectCardProps) {
  return (
    <Link
      to={`/projects/${project.id}`}
      className={`block bg-white border border-gray-200 rounded-2xl shadow-sm hover:shadow-md hover:border-blue-300 transition-all ${
        compact ? 'px-4 py-3' : 'px-4 py-4'
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-base">🗂️</span>
            <span className="font-semibold text-gray-900 truncate">{project.repo_name}</span>
            <StatusBadge status={project.status} />
            {alreadyExists && (
              <span className="text-xs text-gray-400 italic">(already indexed)</span>
            )}
            {indexing && !alreadyExists && (
              <span className="text-xs text-blue-500 italic">indexing started</span>
            )}
          </div>
          <div className="mt-1 text-xs text-gray-500 flex items-center gap-3">
            <span>{project.file_count} files</span>
            <span>·</span>
            <span>{project.node_count} nodes</span>
            {project.progress_pct > 0 && project.progress_pct < 100 && (
              <>
                <span>·</span>
                <span>{project.progress_pct}%</span>
              </>
            )}
          </div>
        </div>
        <span className="text-blue-600 text-sm">→</span>
      </div>
    </Link>
  )
}
