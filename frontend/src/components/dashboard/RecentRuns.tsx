'use client'

import { useQuery } from '@tanstack/react-query'
import { getRecentPipelineRuns } from '@/lib/api'
import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'

export function RecentRuns() {
  const { data: runs, isLoading } = useQuery({
    queryKey: ['recent-runs'],
    queryFn: () => getRecentPipelineRuns(5),
    refetchInterval: 10000,
  })

  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-secondary-200 rounded w-32 mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-secondary-200 rounded-full"></div>
                <div className="flex-1">
                  <div className="h-4 bg-secondary-200 rounded w-24 mb-1"></div>
                  <div className="h-3 bg-secondary-200 rounded w-16"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const recentRuns = runs || []

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-secondary-900">
          Recent Runs
        </h2>
        <Link 
          href="/pipelines" 
          className="text-primary-600 hover:text-primary-700 text-sm font-medium"
        >
          View All
        </Link>
      </div>

      {recentRuns.length === 0 ? (
        <div className="text-center py-6">
          <div className="text-secondary-400 text-sm">
            No pipeline runs yet
          </div>
          <div className="text-secondary-600 text-xs mt-1">
            Start your first content pipeline to see activity here
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {recentRuns.map((run: any) => (
            <Link
              key={run.id}
              href={`/pipelines/${run.id}`}
              className="block p-3 rounded-lg border border-secondary-200 hover:border-primary-300 hover:bg-primary-50 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(run.status)}`} />
                  <div>
                    <div className="font-medium text-secondary-900 text-sm">
                      {run.domain_focus?.join(', ') || 'Content Pipeline'}
                    </div>
                    <div className="text-secondary-600 text-xs">
                      {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={`status-${run.status}`}>
                    {run.status}
                  </span>
                  {run.progress_percentage > 0 && (
                    <div className="text-xs text-secondary-600">
                      {Math.round(run.progress_percentage)}%
                    </div>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

function getStatusColor(status: string) {
  switch (status) {
    case 'running':
      return 'bg-blue-500'
    case 'completed':
      return 'bg-green-500'
    case 'failed':
      return 'bg-red-500'
    case 'paused':
      return 'bg-yellow-500'
    default:
      return 'bg-secondary-400'
  }
}