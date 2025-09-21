'use client'

import { useState, useEffect } from 'react'
import { Header } from '@/components/layout/Header'
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
}

export default function ContentPage() {
  const [content, setContent] = useState<ContentDraft[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedContent, setSelectedContent] = useState<ContentDraft | null>(null)

  useEffect(() => {
    fetchContent()
  }, [])

  const fetchContent = async () => {
    try {
      setLoading(true)
      // Get all completed pipeline runs first
      const runs = await api.get('/pipeline/runs?status=completed')
      const allContent: ContentDraft[] = []
      
      // Fetch content drafts for each completed run
      for (const run of runs.data) {
        try {
          const contentResponse = await api.get(`/content/drafts/${run.id}`)
          if (contentResponse.data && Array.isArray(contentResponse.data)) {
            allContent.push(...contentResponse.data)
          }
        } catch (err) {
          // Skip runs that don't have content yet
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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
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
          <h1 className="text-3xl font-bold text-secondary-900">Generated Content</h1>
          <p className="mt-2 text-secondary-600">
            View and manage content created by your pipelines
          </p>
        </div>

        {/* Content List */}
        {content.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-secondary-500 text-lg">
              No content generated yet. Start a pipeline to create your first content.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Content List Panel */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-secondary-200">
                <h2 className="text-xl font-semibold text-secondary-900">
                  Content Library ({content.length})
                </h2>
              </div>
              <div className="divide-y divide-secondary-200">
                {content.map((item) => (
                  <div 
                    key={item.id}
                    className={`p-4 cursor-pointer hover:bg-secondary-50 transition-colors ${
                      selectedContent?.id === item.id ? 'bg-primary-50 border-r-4 border-primary-500' : ''
                    }`}
                    onClick={() => setSelectedContent(item)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="text-lg font-medium text-secondary-900 mb-1">
                          {item.title}
                        </h3>
                        {item.subtitle && (
                          <p className="text-secondary-600 text-sm mb-2">{item.subtitle}</p>
                        )}
                        <div className="flex items-center space-x-4 text-xs text-secondary-500">
                          <span>{item.word_count} words</span>
                          <span>{item.reading_time_minutes} min read</span>
                          <span>Stage: {item.stage}</span>
                          <span>v{item.version}</span>
                        </div>
                        <div className="mt-2 text-xs text-secondary-400">
                          Created: {formatDate(item.created_at)}
                        </div>
                      </div>
                      {item.readability_score && (
                        <div className="ml-4">
                          <div className="text-xs text-secondary-500">Readability</div>
                          <div className="text-sm font-medium text-secondary-900">
                            {item.readability_score.toFixed(1)}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Content Preview Panel */}
            <div className="bg-white rounded-lg shadow">
              {selectedContent ? (
                <div className="p-6">
                  <div className="border-b border-secondary-200 pb-4 mb-6">
                    <h2 className="text-2xl font-bold text-secondary-900 mb-2">
                      {selectedContent.title}
                    </h2>
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
                    <h4 className="font-medium text-secondary-900 mb-2">Content Preview</h4>
                    <div className="prose max-w-none text-secondary-700 text-sm max-h-96 overflow-y-auto border border-secondary-200 rounded p-4">
                      <pre className="whitespace-pre-wrap font-sans">
                        {selectedContent.content.substring(0, 2000)}
                        {selectedContent.content.length > 2000 && '...'}
                      </pre>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-6 text-center text-secondary-500">
                  Select a content item from the list to preview it here
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}