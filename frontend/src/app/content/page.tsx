'use client'

import { useState, useEffect } from 'react'
import { Header } from '@/components/layout/Header'
import { useSearchParams } from 'next/navigation'
import api from '@/lib/api'

interface ContentDraft {
  id: string
  pipeline_run_id: string
  title: string
  subtitle?: string
  abstract?: string
  content: string
  word_count: number
  reading_time_minutes: number
  readability_score?: number
  meta_description?: string
  keywords: string[]
  tags?: string[]
  stage: string
  version: number
  created_at: string
  review_status: 'draft' | 'approved' | 'rejected' | 'published'
  reviewed_at?: string
  review_notes?: string
  published_at?: string
}

export default function ContentPage() {
  const searchParams = useSearchParams()
  const [content, setContent] = useState<ContentDraft[]>([])
  const [filteredContent, setFilteredContent] = useState<ContentDraft[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedContent, setSelectedContent] = useState<ContentDraft | null>(null)
  const [activeFilter, setActiveFilter] = useState<string>('all')
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  
  // Check URL params for filtering
  const filterParam = searchParams?.get('filter')
  const highlightParam = searchParams?.get('highlight')

  useEffect(() => {
    fetchContent()
  }, [])

  useEffect(() => {
    // Set initial filter from URL params
    if (filterParam === 'pending') {
      setActiveFilter('pending')
    }
  }, [filterParam])

  useEffect(() => {
    // Filter content based on active filter
    let filtered = content
    
    switch (activeFilter) {
      case 'pending':
        filtered = content.filter(item => item.review_status === 'draft')
        break
      case 'approved':
        filtered = content.filter(item => item.review_status === 'approved')
        break
      case 'rejected':
        filtered = content.filter(item => item.review_status === 'rejected')
        break
      case 'published':
        filtered = content.filter(item => item.review_status === 'published')
        break
      default:
        filtered = content
    }
    
    setFilteredContent(filtered)
  }, [content, activeFilter])

  const fetchContent = async () => {
    try {
      setLoading(true)
      
      // Get all pipeline runs that might have content
      const runs = await api.get('/pipeline/runs')
      const allContent: ContentDraft[] = []
      
      // Fetch content drafts for each run that might have content
      for (const run of runs.data) {
        if (!['completed', 'running', 'human_review', 'paused'].includes(run.status)) {
          continue
        }
        
        try {
          const contentResponse = await api.get(`/content/drafts/${run.id}`)
          if (contentResponse.data && Array.isArray(contentResponse.data)) {
            allContent.push(...contentResponse.data)
          }
        } catch (err) {
          console.log(`No content found for run ${run.id}`)
        }
      }
      
      // Sort by creation date, newest first
      allContent.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      setContent(allContent)
      
    } catch (err) {
      setError('Failed to load content')
      console.error('Error fetching content:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleApprove = async (contentId: string, reviewNotes?: string) => {
    try {
      setActionLoading(contentId)
      await api.post(`/content/${contentId}/approve`, { review_notes: reviewNotes })
      await fetchContent() // Refresh content
    } catch (err) {
      console.error('Failed to approve content:', err)
      alert('Failed to approve content')
    } finally {
      setActionLoading(null)
    }
  }

  const handleReject = async (contentId: string, reviewNotes: string, requiredChanges?: string[]) => {
    try {
      setActionLoading(contentId)
      await api.post(`/content/${contentId}/reject`, { 
        review_notes: reviewNotes,
        required_changes: requiredChanges 
      })
      await fetchContent() // Refresh content
    } catch (err) {
      console.error('Failed to reject content:', err)
      alert('Failed to reject content')
    } finally {
      setActionLoading(null)
    }
  }

  const handleDelete = async (contentId: string) => {
    if (!confirm('Are you sure you want to delete this content? This action cannot be undone.')) {
      return
    }
    
    try {
      setActionLoading(contentId)
      await api.delete(`/content/${contentId}`)
      await fetchContent() // Refresh content
      if (selectedContent?.id === contentId) {
        setSelectedContent(null)
      }
    } catch (err) {
      console.error('Failed to delete content:', err)
      alert('Failed to delete content')
    } finally {
      setActionLoading(null)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'draft':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-800 animate-pulse">DRAFT</span>
      case 'approved':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">APPROVED</span>
      case 'rejected':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">REJECTED</span>
      case 'published':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">PUBLISHED</span>
      default:
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">{status.toUpperCase()}</span>
    }
  }

  const getActionButtons = (item: ContentDraft) => {
    const isLoading = actionLoading === item.id
    
    switch (item.review_status) {
      case 'draft':
        return (
          <div className="flex space-x-2">
            <button
              onClick={() => handleApprove(item.id)}
              disabled={isLoading}
              className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
            >
              {isLoading ? '...' : 'Approve'}
            </button>
            <button
              onClick={() => {
                const notes = prompt('Rejection reason (optional):')
                if (notes !== null) handleReject(item.id, notes)
              }}
              disabled={isLoading}
              className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 disabled:opacity-50"
            >
              Reject
            </button>
          </div>
        )
      case 'rejected':
        return (
          <div className="flex space-x-2">
            <button
              onClick={() => window.open(`/content/${item.id}/edit`, '_blank')}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
            >
              Edit
            </button>
          </div>
        )
      case 'approved':
      case 'published':
        return (
          <div className="flex space-x-2">
            <button
              onClick={() => setSelectedContent(item)}
              className="px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-700"
            >
              View
            </button>
          </div>
        )
      default:
        return null
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-center">
            <div className="text-lg">Loading content...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-red-600 text-center">{error}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-secondary-50">
      <Header />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-secondary-900">Content Management</h1>
          <p className="mt-2 text-secondary-600">
            Review, approve, and manage content created by your pipelines
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { key: 'all', label: 'All', count: content.length },
                { key: 'pending', label: 'Pending Review', count: content.filter(c => c.review_status === 'draft').length },
                { key: 'approved', label: 'Approved', count: content.filter(c => c.review_status === 'approved').length },
                { key: 'rejected', label: 'Rejected', count: content.filter(c => c.review_status === 'rejected').length },
                { key: 'published', label: 'Published', count: content.filter(c => c.review_status === 'published').length },
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveFilter(tab.key)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeFilter === tab.key
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label} ({tab.count})
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Content List */}
        {filteredContent.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-secondary-500 text-lg">
              {content.length === 0 
                ? 'No content generated yet. Start a pipeline to create your first content.'
                : `No ${activeFilter === 'all' ? '' : activeFilter} content found.`
              }
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {filteredContent.map((item) => (
              <div 
                key={item.id}
                className={`bg-white rounded-lg shadow p-6 transition-all ${
                  highlightParam === 'true' && item.review_status === 'draft' 
                    ? 'ring-2 ring-orange-400 ring-opacity-75 animate-pulse' 
                    : ''
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      {getStatusBadge(item.review_status)}
                      <h3 className="text-lg font-medium text-secondary-900">
                        {item.title}
                      </h3>
                    </div>
                    
                    {item.abstract && (
                      <p className="text-secondary-600 mb-3 line-clamp-2">{item.abstract}</p>
                    )}
                    
                    <div className="flex items-center space-x-6 text-sm text-secondary-500 mb-3">
                      <span>{item.word_count} words</span>
                      <span>{item.reading_time_minutes} min read</span>
                      <span>Stage: {item.stage}</span>
                      <span>Created: {formatDate(item.created_at)}</span>
                    </div>
                    
                    {item.review_notes && (
                      <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-700">
                          <strong>Review Notes:</strong> {item.review_notes}
                        </p>
                      </div>
                    )}
                  </div>
                  
                  <div className="ml-6 flex flex-col space-y-2">
                    {getActionButtons(item)}
                    <button
                      onClick={() => setSelectedContent(item)}
                      className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded hover:bg-gray-200"
                    >
                      View
                    </button>
                    <button
                      onClick={() => handleDelete(item.id)}
                      disabled={actionLoading === item.id}
                      className="px-3 py-1 bg-red-100 text-red-700 text-sm rounded hover:bg-red-200 disabled:opacity-50"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Content Preview Modal */}
        {selectedContent && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-start mb-6">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      {getStatusBadge(selectedContent.review_status)}
                      <h2 className="text-2xl font-bold text-secondary-900">
                        {selectedContent.title}
                      </h2>
                    </div>
                    {selectedContent.subtitle && (
                      <p className="text-lg text-secondary-600 mb-2">{selectedContent.subtitle}</p>
                    )}
                    <div className="flex items-center space-x-6 text-sm text-secondary-500">
                      <span>{selectedContent.word_count} words</span>
                      <span>{selectedContent.reading_time_minutes} minutes</span>
                      {selectedContent.readability_score && (
                        <span>Readability: {selectedContent.readability_score.toFixed(1)}</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedContent(null)}
                    className="ml-4 text-gray-400 hover:text-gray-600"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Review Information */}
                {(selectedContent.reviewed_at || selectedContent.review_notes) && (
                  <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <h4 className="font-medium text-secondary-900 mb-2">Review Information</h4>
                    {selectedContent.reviewed_at && (
                      <p className="text-sm text-secondary-600 mb-1">
                        Reviewed: {formatDate(selectedContent.reviewed_at)}
                      </p>
                    )}
                    {selectedContent.review_notes && (
                      <p className="text-sm text-secondary-700">
                        <strong>Notes:</strong> {selectedContent.review_notes}
                      </p>
                    )}
                  </div>
                )}

                {/* Meta Description */}
                {selectedContent.meta_description && (
                  <div className="mb-6 p-4 bg-secondary-50 rounded-lg">
                    <h4 className="font-medium text-secondary-900 mb-2">Meta Description</h4>
                    <p className="text-secondary-600 text-sm">{selectedContent.meta_description}</p>
                  </div>
                )}

                {/* Keywords & Tags */}
                <div className="mb-6">
                  {selectedContent.keywords && selectedContent.keywords.length > 0 && (
                    <div className="mb-4">
                      <h4 className="font-medium text-secondary-900 mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedContent.keywords.map((keyword, idx) => (
                          <span 
                            key={idx}
                            className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {selectedContent.tags && selectedContent.tags.length > 0 && (
                    <div>
                      <h4 className="font-medium text-secondary-900 mb-2">Tags</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedContent.tags.map((tag, idx) => (
                          <span 
                            key={idx}
                            className="px-3 py-1 bg-secondary-100 text-secondary-700 rounded-full text-sm"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Abstract */}
                {selectedContent.abstract && (
                  <div className="mb-6">
                    <h4 className="font-medium text-secondary-900 mb-2">Abstract</h4>
                    <p className="text-secondary-600">{selectedContent.abstract}</p>
                  </div>
                )}

                {/* Content Preview */}
                <div>
                  <h4 className="font-medium text-secondary-900 mb-2">Full Content</h4>
                  <div className="prose max-w-none text-secondary-700 text-sm border border-secondary-200 rounded p-4 bg-white max-h-96 overflow-y-auto">
                    <pre className="whitespace-pre-wrap font-sans">
                      {selectedContent.content}
                    </pre>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="mt-6 flex justify-end space-x-3">
                  {getActionButtons(selectedContent)}
                  <button
                    onClick={() => setSelectedContent(null)}
                    className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}