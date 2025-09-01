'use client'

import { useQuery } from '@tanstack/react-query'
import { getPipelineStats } from '@/lib/api'

export function PipelineOverview() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['pipeline-stats'],
    queryFn: getPipelineStats,
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  if (isLoading) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-6 bg-secondary-200 rounded w-48 mb-4"></div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i}>
                <div className="h-8 bg-secondary-200 rounded mb-2"></div>
                <div className="h-4 bg-secondary-200 rounded w-16"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const defaultStats = {
    total_runs: 0,
    active_runs: 0,
    completed_runs: 0,
    failed_runs: 0,
    success_rate: 0,
    avg_processing_time: 0
  }

  const pipelineStats = stats || defaultStats

  return (
    <div className="card">
      <h2 className="text-xl font-semibold text-secondary-900 mb-4">
        Pipeline Overview
      </h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <div className="text-2xl font-bold text-primary-600">
            {pipelineStats.total_runs}
          </div>
          <div className="text-sm text-secondary-600">Total Runs</div>
        </div>
        
        <div>
          <div className="text-2xl font-bold text-blue-600">
            {pipelineStats.active_runs}
          </div>
          <div className="text-sm text-secondary-600">Active</div>
        </div>
        
        <div>
          <div className="text-2xl font-bold text-green-600">
            {pipelineStats.completed_runs}
          </div>
          <div className="text-sm text-secondary-600">Completed</div>
        </div>
        
        <div>
          <div className="text-2xl font-bold text-red-600">
            {pipelineStats.failed_runs}
          </div>
          <div className="text-sm text-secondary-600">Failed</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="text-lg font-semibold text-secondary-900">
            {(pipelineStats.success_rate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-secondary-600">Success Rate</div>
        </div>
        
        <div>
          <div className="text-lg font-semibold text-secondary-900">
            {Math.round(pipelineStats.avg_processing_time)}m
          </div>
          <div className="text-sm text-secondary-600">Avg. Processing Time</div>
        </div>
      </div>
    </div>
  )
}