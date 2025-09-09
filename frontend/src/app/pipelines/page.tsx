'use client'

import { useQuery } from '@tanstack/react-query'
import { getPipelineRuns } from '@/lib/api'
import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { useState } from 'react'

export default function PipelinesPage() {
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [limit, setLimit] = useState(20)

  const { data: runs = [], isLoading, error } = useQuery({
    queryKey: ['pipeline-runs', statusFilter, limit],
    queryFn: () => getPipelineRuns({ 
      limit, 
      status: statusFilter || undefined 
    }),
    refetchInterval: 10000,
  })

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-6xl mx-auto">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="text-red-800 font-medium">Error Loading Pipeline Runs</h3>
            <p className="text-red-600 text-sm mt-1">
              {error instanceof Error ? error.message : 'Failed to load pipeline runs'}
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Pipeline Runs
          </h1>
          <p className="text-gray-600">
            View and manage all your content pipeline executions
          </p>
        </div>

        {/* Filters */}
        <div className="card mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div>
                <label htmlFor="status-filter" className="block text-sm font-medium text-gray-700 mb-1">
                  Filter by Status
                </label>
                <select
                  id="status-filter"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="">All Statuses</option>
                  <option value="initialized">Initialized</option>
                  <option value="running">Running</option>
                  <option value="paused">Paused</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
            </div>
            
            <Link
              href="/"
              className="btn-primary"
            >
              New Pipeline
            </Link>
          </div>
        </div>

        {/* Pipeline Runs List */}
        <div className="card">
          {isLoading ? (
            <div className="animate-pulse space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex items-center space-x-4 p-4 border-b border-gray-200">
                  <div className="w-4 h-4 bg-gray-200 rounded-full"></div>
                  <div className="flex-1">
                    <div className="h-5 bg-gray-200 rounded w-1/3 mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                  </div>
                  <div className="w-20 h-6 bg-gray-200 rounded"></div>
                </div>
              ))}
            </div>
          ) : runs.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-gray-400 text-lg mb-2">
                No pipeline runs found
              </div>
              <p className="text-gray-600 mb-6">
                {statusFilter 
                  ? `No runs with status "${statusFilter}"`
                  : "Start your first content pipeline to see runs here"
                }
              </p>
              <Link
                href="/"
                className="btn-primary"
              >
                Start New Pipeline
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {runs.map((run: any) => (
                <Link
                  key={run.id}
                  href={`/pipelines/${run.id}`}
                  className="block p-6 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`w-4 h-4 rounded-full ${getStatusColor(run.status)}`} />
                      <div>
                        <div className="font-medium text-gray-900 text-lg">
                          {run.domain_focus?.join(', ') || 'Content Pipeline'}
                        </div>
                        <div className="text-gray-600 text-sm mt-1">
                          Created {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                        </div>
                        {run.current_step && (
                          <div className="text-gray-500 text-xs mt-1">
                            Step: {run.current_step}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      {run.progress_percentage > 0 && (
                        <div className="text-right">
                          <div className="text-sm font-medium text-gray-900">
                            {Math.round(run.progress_percentage)}%
                          </div>
                          <div className="w-24 bg-gray-200 rounded-full h-2 mt-1">
                            <div 
                              className="bg-primary-600 h-2 rounded-full" 
                              style={{ width: `${run.progress_percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      
                      <div className="text-right">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusStyle(run.status)}`}>
                          {run.status}
                        </span>
                        {run.final_quality_score && (
                          <div className="text-xs text-gray-600 mt-1">
                            Quality: {Math.round(run.final_quality_score * 100)}%
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {run.error_message && (
                    <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
                      <div className="text-red-800 text-sm font-medium">Error</div>
                      <div className="text-red-600 text-xs mt-1">{run.error_message}</div>
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>

        {runs.length >= limit && (
          <div className="mt-6 text-center">
            <button
              onClick={() => setLimit(limit + 20)}
              className="btn-secondary"
            >
              Load More
            </button>
          </div>
        )}
      </div>
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
    case 'cancelled':
      return 'bg-gray-500'
    default:
      return 'bg-gray-400'
  }
}

function getStatusStyle(status: string) {
  switch (status) {
    case 'running':
      return 'bg-blue-100 text-blue-800'
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'failed':
      return 'bg-red-100 text-red-800'
    case 'paused':
      return 'bg-yellow-100 text-yellow-800'
    case 'cancelled':
      return 'bg-gray-100 text-gray-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}