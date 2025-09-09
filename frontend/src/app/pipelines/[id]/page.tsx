'use client'

import { useQuery } from '@tanstack/react-query'
import { getPipelineRun } from '@/lib/api'
import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { useParams } from 'next/navigation'

export default function PipelineRunPage() {
  const params = useParams()
  const runId = params.id as string

  const { data: run, isLoading, error } = useQuery({
    queryKey: ['pipeline-run', runId],
    queryFn: () => getPipelineRun(runId),
    refetchInterval: 5000,
    enabled: !!runId,
  })

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="text-red-800 font-medium">Error Loading Pipeline Run</h3>
            <p className="text-red-600 text-sm mt-1">
              {error instanceof Error ? error.message : 'Failed to load pipeline run'}
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
            <div className="card mb-6">
              <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
              <div className="space-y-3">
                <div className="h-4 bg-gray-200 rounded w-full"></div>
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!run) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Pipeline Run Not Found
            </h2>
            <p className="text-gray-600 mb-6">
              The pipeline run you're looking for doesn't exist or has been deleted.
            </p>
            <Link href="/pipelines" className="btn-primary">
              Back to Pipeline Runs
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Link href="/pipelines" className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            ← Back to Pipeline Runs
          </Link>
        </div>

        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <div className={`w-6 h-6 rounded-full ${getStatusColor(run.status)}`} />
            <h1 className="text-3xl font-bold text-gray-900">
              {run.domain_focus?.join(', ') || 'Content Pipeline'}
            </h1>
          </div>
          <p className="text-gray-600">
            Pipeline Run #{run.id.slice(0, 8)}
          </p>
        </div>

        {/* Status Overview */}
        <div className="card mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Status Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="text-sm font-medium text-gray-500">Status</div>
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium mt-1 ${getStatusStyle(run.status)}`}>
                {run.status}
              </span>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-500">Progress</div>
              <div className="mt-1">
                <div className="text-lg font-semibold text-gray-900">
                  {Math.round(run.progress_percentage)}%
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div 
                    className="bg-primary-600 h-2 rounded-full" 
                    style={{ width: `${run.progress_percentage}%` }}
                  ></div>
                </div>
              </div>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-500">Current Step</div>
              <div className="text-lg font-semibold text-gray-900 mt-1">
                {run.current_step || 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline Details */}
        <div className="card mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Pipeline Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="text-sm font-medium text-gray-500 mb-2">Domain Focus</div>
              <div className="flex flex-wrap gap-2">
                {run.domain_focus?.map((domain: string, index: number) => (
                  <span key={index} className="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm">
                    {domain}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-500 mb-2">Quality Thresholds</div>
              <div className="space-y-1 text-sm">
                {Object.entries(run.quality_thresholds || {}).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="capitalize">{key.replace('_', ' ')}:</span>
                    <span className="font-medium">{Math.round(value * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Timeline */}
        <div className="card mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Timeline</h2>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <div>
                <div className="font-medium">Created</div>
                <div className="text-sm text-gray-600">
                  {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                </div>
              </div>
            </div>
            
            {run.started_at && (
              <div className="flex items-center space-x-4">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div>
                  <div className="font-medium">Started</div>
                  <div className="text-sm text-gray-600">
                    {formatDistanceToNow(new Date(run.started_at), { addSuffix: true })}
                  </div>
                </div>
              </div>
            )}
            
            {run.completed_at && (
              <div className="flex items-center space-x-4">
                <div className={`w-2 h-2 rounded-full ${run.status === 'completed' ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <div>
                  <div className="font-medium">
                    {run.status === 'completed' ? 'Completed' : 'Ended'}
                  </div>
                  <div className="text-sm text-gray-600">
                    {formatDistanceToNow(new Date(run.completed_at), { addSuffix: true })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Quality Score */}
        {run.final_quality_score && (
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Quality Assessment</h2>
            <div className="flex items-center space-x-4">
              <div className="text-3xl font-bold text-gray-900">
                {Math.round(run.final_quality_score * 100)}%
              </div>
              <div>
                <div className="font-medium">Final Quality Score</div>
                <div className="text-sm text-gray-600">
                  {run.final_quality_score >= 0.85 ? 'Meets quality threshold' : 'Below quality threshold'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Published URLs */}
        {run.published_urls && run.published_urls.length > 0 && (
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Published Content</h2>
            <div className="space-y-2">
              {run.published_urls.map((url: string, index: number) => (
                <a 
                  key={index}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors"
                >
                  <div className="text-primary-600 hover:text-primary-700 font-medium">
                    {url}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Click to view published content
                  </div>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Error Message */}
        {run.error_message && (
          <div className="card border-red-200 bg-red-50">
            <h2 className="text-xl font-semibold text-red-800 mb-4">Error Details</h2>
            <div className="text-red-700 bg-red-100 p-3 rounded-md">
              {run.error_message}
            </div>
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