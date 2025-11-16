'use client'

import { useState, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { 
  getAgentLogs, 
  deleteAgentLogs, 
  getPipelineTopics,
  getContentOutlines,
  getQualityAssessments,
  getPublications,
  type AgentLog 
} from '@/lib/api'
import { formatDistanceToNow } from 'date-fns'

// Agent context renderer for enhanced log display
interface AgentContextRendererProps {
  context: Record<string, any>
  agentName: string
  stage: string
  pipelineRunId: string
}

function AgentContextRenderer({ context, agentName, stage, pipelineRunId }: AgentContextRendererProps) {
  const [showRawJson, setShowRawJson] = useState(false)
  const [showDetailsModal, setShowDetailsModal] = useState<string | null>(null)
  
  // Queries for detail data (only fetch when needed)
  const { data: topics, isLoading: loadingTopics } = useQuery({
    queryKey: ['topics', pipelineRunId],
    queryFn: () => getPipelineTopics(pipelineRunId),
    enabled: showDetailsModal === 'topics'
  })
  
  const { data: outlines, isLoading: loadingOutlines } = useQuery({
    queryKey: ['outlines', pipelineRunId],
    queryFn: () => getContentOutlines(pipelineRunId),
    enabled: showDetailsModal === 'outlines'
  })
  
  const { data: assessments, isLoading: loadingAssessments } = useQuery({
    queryKey: ['assessments', pipelineRunId],
    queryFn: () => getQualityAssessments(pipelineRunId),
    enabled: showDetailsModal === 'assessments'
  })
  
  const { data: publications, isLoading: loadingPublications } = useQuery({
    queryKey: ['publications', pipelineRunId],
    queryFn: () => getPublications(pipelineRunId),
    enabled: showDetailsModal === 'publications'
  })
  
  // Detect pointer keys for enhanced rendering
  const hasPointerKeys = () => {
    return !!(
      context.topic_ids || context.source_ids || context.chosen_topic_id ||
      context.outline_id || context.content_draft_id || context.quality_assessment_ids ||
      context.critique_report_id || context.channel_content_ids || context.human_review_id ||
      context.publication_ids
    )
  }

  // Render enhanced content based on agent type and pointer keys
  const renderEnhancedContent = () => {
    // Research Agent - Topics and Sources
    if (agentName === 'ResearchCoordinatorAgent' && (context.topic_ids || context.source_ids)) {
      return (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900">Research Results</h4>
            <span className="text-xs text-gray-500">{context.topics_count || 0} topics, {context.sources_count || 0} sources</span>
          </div>
          <div className="text-sm text-gray-600">
            <p>• {context.topics_count || 0} research topics generated</p>
            <p>• {context.sources_count || 0} credible sources found</p>
            {context.topic_ids?.length > 0 && (
              <button 
                onClick={(e) => { e.stopPropagation(); setShowDetailsModal('topics'); }}
                className="mt-2 text-blue-600 hover:text-blue-800 text-xs underline"
              >
                View Topics Detail
              </button>
            )}
          </div>
        </div>
      )
    }

    // Curation Agent - Selected Topic
    if (agentName === 'ContentCuratorAgent' && context.chosen_topic_id) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Topic Selection</h4>
          <div className="text-sm text-gray-600">
            <p>• Topic selected: {context.chosen_topic_id}</p>
            <button 
              onClick={(e) => { e.stopPropagation(); setShowDetailsModal('topics'); }}
              className="mt-2 text-blue-600 hover:text-blue-800 text-xs underline"
            >
              View Selected Topic
            </button>
          </div>
        </div>
      )
    }

    // SEO Strategy Agent - Outline
    if (agentName === 'SEOStrategistAgent' && context.outline_id) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">SEO Outline</h4>
          <div className="text-sm text-gray-600">
            <p>• {context.sections_count || 0} sections structured</p>
            <p>• SEO optimization applied</p>
            <button 
              onClick={(e) => { e.stopPropagation(); setShowDetailsModal('outlines'); }}
              className="mt-2 text-blue-600 hover:text-blue-800 text-xs underline"
            >
              View Outline Structure
            </button>
          </div>
        </div>
      )
    }

    // Writing Agent - Content Draft
    if (agentName === 'ContentWriterAgent' && context.content_draft_id) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Content Draft</h4>
          <div className="text-sm text-gray-600">
            <p>• {context.word_count || 0} words generated</p>
            <p>• Version: {context.version || 'N/A'}</p>
            <p>• Stage: {context.stage || 'draft'}</p>
            <a 
              href={`/pipelines/${pipelineRunId}?tab=content&draftId=${context.content_draft_id}`}
              onClick={(e) => e.stopPropagation()}
              className="mt-2 inline-block text-blue-600 hover:text-blue-800 text-xs underline"
            >
              View in Content Tab →
            </a>
          </div>
        </div>
      )
    }

    // Quality Gates - Assessment Results
    if (agentName === 'QualityGateAgents' && context.quality_assessment_ids) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Quality Assessment</h4>
          <div className="text-sm text-gray-600">
            <p>• Overall Score: {(context.overall_score * 100).toFixed(1)}%</p>
            <div className="flex flex-wrap gap-1 mt-2">
              {Object.keys(context.quality_assessment_ids).map(gate => (
                <span key={gate} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                  {gate.replace('_', ' ')}
                </span>
              ))}
            </div>
            <button 
              onClick={(e) => { e.stopPropagation(); setShowDetailsModal('assessments'); }}
              className="mt-2 text-blue-600 hover:text-blue-800 text-xs underline"
            >
              View Assessment Details
            </button>
          </div>
        </div>
      )
    }

    // Human Review Agent
    if (agentName === 'HumanReviewGateAgent' && context.content_draft_id) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Human Review</h4>
          <div className="text-sm text-gray-600">
            <p>• Status: {context.review_status || 'pending'}</p>
            <a 
              href={`/pipelines/${pipelineRunId}?tab=content&draftId=${context.content_draft_id}`}
              onClick={(e) => e.stopPropagation()}
              className="mt-2 inline-block text-blue-600 hover:text-blue-800 text-xs underline"
            >
              Review Content →
            </a>
          </div>
        </div>
      )
    }

    // Publisher Agent - Published URLs
    if (agentName === 'PublisherAgent' && context.published_urls) {
      return (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Publication Results</h4>
          <div className="text-sm text-gray-600">
            <p>• {context.published_urls.length} URLs published</p>
            {context.published_urls.slice(0, 2).map((url: string, idx: number) => (
              <a key={idx} href={url} target="_blank" rel="noopener noreferrer" 
                 onClick={(e) => e.stopPropagation()}
                 className="block text-blue-600 hover:text-blue-800 text-xs underline truncate">
                {url}
              </a>
            ))}
            {context.published_urls.length > 2 && (
              <p className="text-xs text-gray-500">... and {context.published_urls.length - 2} more</p>
            )}
          </div>
        </div>
      )
    }

    return null
  }

  const enhancedContent = renderEnhancedContent()

  return (
    <div className="space-y-3">
      {/* Enhanced Content (if available) */}
      {enhancedContent && !showRawJson && enhancedContent}
      
      {/* Raw JSON Toggle */}
      <div className="flex items-center justify-between">
        <button
          onClick={(e) => { e.stopPropagation(); setShowRawJson(!showRawJson); }}
          className="text-xs text-gray-500 hover:text-gray-700 underline"
        >
          {showRawJson ? 'Show Enhanced View' : 'Show Raw JSON'}
        </button>
        {hasPointerKeys() && !showRawJson && (
          <span className="text-xs text-green-600">✨ Enhanced view available</span>
        )}
      </div>
      
      {/* Raw JSON (fallback or toggle) */}
      {(showRawJson || !enhancedContent) && (
        <pre className="text-xs text-gray-600 overflow-auto max-h-40 bg-gray-100 p-2 rounded">
          {JSON.stringify(context, null, 2)}
        </pre>
      )}

      {/* Simple Details Modal */}
      {showDetailsModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowDetailsModal(null)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl max-h-[80vh] overflow-auto" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                {showDetailsModal === 'topics' && 'Research Topics'}
                {showDetailsModal === 'outlines' && 'Content Outline'}
                {showDetailsModal === 'assessments' && 'Quality Assessments'}
                {showDetailsModal === 'publications' && 'Publications'}
              </h3>
              <button onClick={() => setShowDetailsModal(null)} className="text-gray-500 hover:text-gray-700">
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              {showDetailsModal === 'topics' && (
                loadingTopics ? (
                  <div className="text-center py-4">Loading topics...</div>
                ) : topics?.length ? (
                  topics.slice(0, 5).map((topic: any) => (
                    <div key={topic.id} className="border rounded p-3">
                      <h4 className="font-medium">{topic.title}</h4>
                      <p className="text-sm text-gray-600 mt-1">{topic.description}</p>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {topic.target_keywords?.slice(0, 3).map((keyword: string) => (
                          <span key={keyword} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-500">No topics found</div>
                )
              )}

              {showDetailsModal === 'outlines' && (
                loadingOutlines ? (
                  <div className="text-center py-4">Loading outlines...</div>
                ) : outlines?.length ? (
                  outlines.map((outline: any) => (
                    <div key={outline.id} className="space-y-2">
                      <h4 className="font-medium">{outline.title}</h4>
                      <div className="space-y-1">
                        {outline.sections?.map((section: any, idx: number) => (
                          <div key={idx} className="text-sm pl-4 border-l-2 border-gray-200">
                            <span className="font-medium">{section.title}</span>
                            {section.estimated_words && (
                              <span className="text-gray-500 ml-2">({section.estimated_words} words)</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-500">No outlines found</div>
                )
              )}

              {showDetailsModal === 'assessments' && (
                loadingAssessments ? (
                  <div className="text-center py-4">Loading assessments...</div>
                ) : assessments ? (
                  <div className="space-y-3">
                    <div className="text-lg font-medium">Overall Score: {(assessments.overall_score * 100).toFixed(1)}%</div>
                    {Object.entries(assessments.assessments_by_gate || {}).map(([gate, assessmentList]: [string, any]) => (
                      <div key={gate} className="border rounded p-3">
                        <h5 className="font-medium capitalize">{gate.replace('_', ' ')}</h5>
                        {Array.isArray(assessmentList) && assessmentList.map((assessment: any) => (
                          <div key={assessment.id} className="mt-2 text-sm">
                            <div className="flex justify-between">
                              <span>Score: {(assessment.score * 100).toFixed(1)}%</span>
                              <span className={assessment.passed ? 'text-green-600' : 'text-red-600'}>
                                {assessment.passed ? 'PASSED' : 'FAILED'}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-4 text-gray-500">No assessments found</div>
                )
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface AgentLogProps {
  pipelineRunId: string
}

const LOG_LEVEL_COLORS = {
  INFO: 'text-green-600 bg-green-50 border-green-200',
  WARNING: 'text-yellow-600 bg-yellow-50 border-yellow-200',
  ERROR: 'text-red-600 bg-red-50 border-red-200',
  DEBUG: 'text-blue-600 bg-blue-50 border-blue-200'
}

const LOG_LEVEL_INDICATORS = {
  INFO: '🟢',
  WARNING: '🟡',
  ERROR: '🔴',
  DEBUG: '🔵'
}

export default function AgentLogComponent({ pipelineRunId }: AgentLogProps) {
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set())
  const [filters, setFilters] = useState({
    level: '',
    agent_name: ''
  })
  const [isDeleting, setIsDeleting] = useState(false)

  const queryClient = useQueryClient()

  const { data: logs = [], isLoading, error, refetch } = useQuery({
    queryKey: ['agent-logs', pipelineRunId, filters],
    queryFn: () => getAgentLogs(pipelineRunId, { 
      limit: 100,
      level: filters.level || undefined,
      agent_name: filters.agent_name || undefined
    }),
    refetchInterval: autoRefresh ? 5000 : false,
    enabled: !!pipelineRunId,
  })

  // Get unique agent names and levels for filters
  const uniqueAgentNames = Array.from(new Set(logs.map(log => log.agent_name)))
  const uniqueLevels = Array.from(new Set(logs.map(log => log.level)))

  const handleDeleteLogs = async () => {
    if (!confirm('Are you sure you want to delete all agent logs for this pipeline? This action cannot be undone.')) {
      return
    }

    setIsDeleting(true)
    try {
      await deleteAgentLogs(pipelineRunId)
      queryClient.invalidateQueries({ queryKey: ['agent-logs', pipelineRunId] })
    } catch (error) {
      console.error('Failed to delete logs:', error)
      alert('Failed to delete logs. Please try again.')
    } finally {
      setIsDeleting(false)
    }
  }

  const toggleLogExpansion = (logId: string) => {
    const newExpanded = new Set(expandedLogs)
    if (newExpanded.has(logId)) {
      newExpanded.delete(logId)
    } else {
      newExpanded.add(logId)
    }
    setExpandedLogs(newExpanded)
  }

  const handleRefresh = () => {
    refetch()
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium">Error Loading Agent Logs</h3>
          <p className="text-red-600 text-sm mt-1">
            {error instanceof Error ? error.message : 'Failed to load agent logs'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6">
      {/* Header with controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div className="flex items-center space-x-4">
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span>🔄</span>
            <span>{isLoading ? 'Refreshing...' : 'Refresh'}</span>
          </button>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm text-gray-700">Auto-refresh</span>
          </label>
        </div>

        <div className="flex items-center space-x-4">
          {/* Filter Controls */}
          <select
            value={filters.level}
            onChange={(e) => setFilters(prev => ({ ...prev, level: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="">All Levels</option>
            {uniqueLevels.map(level => (
              <option key={level} value={level}>{level}</option>
            ))}
          </select>

          <select
            value={filters.agent_name}
            onChange={(e) => setFilters(prev => ({ ...prev, agent_name: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="">All Agents</option>
            {uniqueAgentNames.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>

          <button
            onClick={handleDeleteLogs}
            disabled={isDeleting || logs.length === 0}
            className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span>🗑️</span>
            <span>{isDeleting ? 'Deleting...' : 'Delete Logs'}</span>
          </button>
        </div>
      </div>

      {/* Logs Display */}
      {isLoading && logs.length === 0 ? (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse border border-gray-200 rounded-lg p-4">
              <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      ) : logs.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 text-6xl mb-4">📄</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Agent Logs</h3>
          <p className="text-gray-600">
            {filters.level || filters.agent_name 
              ? 'No logs match your current filters.' 
              : 'No agent logs have been recorded for this pipeline yet.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {logs.map((log) => (
            <div
              key={log.id}
              className={`border rounded-lg p-4 transition-all cursor-pointer hover:shadow-md ${
                LOG_LEVEL_COLORS[log.level] || LOG_LEVEL_COLORS.INFO
              }`}
              onClick={() => toggleLogExpansion(log.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="text-lg">{LOG_LEVEL_INDICATORS[log.level]}</span>
                    <span className="text-sm text-gray-500">
                      {formatDistanceToNow(new Date(log.timestamp), { addSuffix: true })}
                    </span>
                    <span className="font-medium text-gray-900">{log.agent_name}</span>
                  </div>
                  
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-medium">STAGE:</span> {log.stage} | 
                    <span className="font-medium ml-2">OPERATION:</span> {log.operation}
                  </div>
                  
                  <div className="text-gray-900">
                    {log.message}
                  </div>
                  
                  {expandedLogs.has(log.id) && log.context && Object.keys(log.context).length > 0 && (
                    <div className="mt-3 p-3 bg-gray-50 rounded border">
                      <div className="text-sm font-medium text-gray-700 mb-2">Context:</div>
                      <AgentContextRenderer 
                        context={log.context}
                        agentName={log.agent_name}
                        stage={log.stage}
                        pipelineRunId={pipelineRunId}
                      />
                    </div>
                  )}
                </div>
                
                <div className="ml-4 flex-shrink-0">
                  <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    LOG_LEVEL_COLORS[log.level] || LOG_LEVEL_COLORS.INFO
                  }`}>
                    {log.level}
                  </div>
                </div>
              </div>
              
              {log.context && Object.keys(log.context).length > 0 && (
                <div className="mt-2 text-xs text-gray-500">
                  {expandedLogs.has(log.id) ? 'Click to collapse context' : 'Click to view context'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Log Summary */}
      {logs.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>
              Showing {logs.length} log{logs.length !== 1 ? 's' : ''}
              {(filters.level || filters.agent_name) && ' (filtered)'}
            </span>
            <span>
              Last updated: {formatDistanceToNow(new Date(), { addSuffix: true })}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}