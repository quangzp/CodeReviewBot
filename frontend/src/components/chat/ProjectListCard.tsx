import type { ProjectSummary } from '../../types'
import ProjectCard from './ProjectCard'

interface ProjectListCardProps {
  projects: ProjectSummary[]
}

export default function ProjectListCard({ projects }: ProjectListCardProps) {
  if (projects.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic px-4 py-3 bg-gray-50 rounded-2xl border border-gray-200">
        No projects yet.
      </div>
    )
  }
  return (
    <div className="space-y-2">
      {projects.map((p) => (
        <ProjectCard key={p.id} project={p} compact />
      ))}
    </div>
  )
}
