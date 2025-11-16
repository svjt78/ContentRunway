'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
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
  meta_description?: string
  keywords: string[]
  tags?: string[]
  review_status: 'draft' | 'approved' | 'rejected' | 'published'
  reviewed_at?: string
  review_notes?: string
}

export default function ContentEditPage() {
  const params = useParams()
  const router = useRouter()
  const contentId = params?.id as string
  
  const [content, setContent] = useState<ContentDraft | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Form state
  const [title, setTitle] = useState('')
  const [contentText, setContentText] = useState('')
  const [metaDescription, setMetaDescription] = useState('')
  const [keywords, setKeywords] = useState<string[]>([])
  const [reviewNotes, setReviewNotes] = useState('')
  const [newKeyword, setNewKeyword] = useState('')

  useEffect(() => {
    if (contentId) {
      fetchContent()
    }
  }, [contentId])

  const fetchContent = async () => {
    try {
      setLoading(true)
      const response = await api.get(`/content/${contentId}`)
      const contentData = response.data
      
      setContent(contentData)
      setTitle(contentData.title)
      setContentText(contentData.content)
      setMetaDescription(contentData.meta_description || '')
      setKeywords(contentData.keywords || [])
      setReviewNotes(contentData.review_notes || '')
      
    } catch (err) {
      setError('Failed to load content for editing')
      console.error('Error fetching content:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleAddKeyword = () => {
    if (newKeyword.trim() && !keywords.includes(newKeyword.trim())) {
      setKeywords([...keywords, newKeyword.trim()])
      setNewKeyword('')
    }
  }

  const handleRemoveKeyword = (keywordToRemove: string) => {
    setKeywords(keywords.filter(k => k !== keywordToRemove))
  }

  const handleSave = async (submitForReview: boolean = false) => {
    try {
      setSaving(true)
      
      await api.put(`/content/${contentId}`, {
        title,
        content: contentText,
        meta_description: metaDescription,
        keywords,
        review_notes: reviewNotes
      })
      
      if (submitForReview) {
        router.push('/content?filter=pending&highlight=true')
      } else {
        alert('Content saved successfully')
      }
      
    } catch (err) {
      console.error('Failed to save content:', err)
      alert('Failed to save content')
    } finally {
      setSaving(false)
    }
  }

  const handleCancel = () => {
    if (confirm('Are you sure you want to cancel? Your changes will be lost.')) {
      router.push('/content')
    }
  }

  // Auto-save every 30 seconds
  useEffect(() => {
    if (!content || loading) return

    const autoSaveInterval = setInterval(() => {
      if (title !== content.title || 
          contentText !== content.content || 
          metaDescription !== (content.meta_description || '') ||
          JSON.stringify(keywords) !== JSON.stringify(content.keywords || [])) {
        handleSave(false)
      }
    }, 30000)

    return () => clearInterval(autoSaveInterval)
  }, [title, contentText, metaDescription, keywords, content, loading])

  if (loading) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-center">
            <div className="text-lg">Loading content for editing...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !content) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-red-600 text-center">{error || 'Content not found'}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-secondary-50">
      <Header />
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <button
                onClick={() => router.push('/content')}
                className="mb-4 text-primary-600 hover:text-primary-700 flex items-center"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Content
              </button>
              <h1 className="text-3xl font-bold text-secondary-900">Content Editor</h1>
              <p className="mt-2 text-secondary-600">
                Edit and improve your content based on review feedback
              </p>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={handleCancel}
                disabled={saving}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleSave(false)}
                disabled={saving}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Draft'}
              </button>
              <button
                onClick={() => handleSave(true)}
                disabled={saving}
                className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
              >
                Save & Submit for Review
              </button>
            </div>
          </div>
        </div>

        {/* Review Feedback */}
        {content.review_notes && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <h3 className="font-medium text-red-900 mb-2">Review Feedback</h3>
            <p className="text-red-700">{content.review_notes}</p>
          </div>
        )}

        {/* Edit Form */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6">
            {/* Title */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Content title..."
              />
            </div>

            {/* Meta Description */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Meta Description
              </label>
              <textarea
                value={metaDescription}
                onChange={(e) => setMetaDescription(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Brief description for SEO..."
              />
            </div>

            {/* Keywords */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Keywords
              </label>
              <div className="flex flex-wrap gap-2 mb-3">
                {keywords.map((keyword, idx) => (
                  <span 
                    key={idx}
                    className="inline-flex items-center px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                  >
                    {keyword}
                    <button
                      onClick={() => handleRemoveKeyword(keyword)}
                      className="ml-2 text-primary-500 hover:text-primary-700"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex">
                <input
                  type="text"
                  value={newKeyword}
                  onChange={(e) => setNewKeyword(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddKeyword()}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  placeholder="Add keyword..."
                />
                <button
                  onClick={handleAddKeyword}
                  className="px-4 py-2 bg-primary-600 text-white rounded-r-md hover:bg-primary-700"
                >
                  Add
                </button>
              </div>
            </div>

            {/* Content Editor */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Content
              </label>
              <textarea
                value={contentText}
                onChange={(e) => setContentText(e.target.value)}
                rows={20}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono text-sm"
                placeholder="Write your content here..."
              />
              <div className="mt-2 text-sm text-gray-500">
                {contentText.split(' ').length} words • {Math.ceil(contentText.split(' ').length / 200)} min read
              </div>
            </div>

            {/* Review Notes */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Review Notes (Optional)
              </label>
              <textarea
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Add notes about your changes..."
              />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <button
                onClick={handleCancel}
                disabled={saving}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleSave(false)}
                disabled={saving}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Draft'}
              </button>
              <button
                onClick={() => handleSave(true)}
                disabled={saving}
                className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
              >
                Save & Submit for Review
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}