'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Header } from '@/components/layout/Header'
import api from '@/lib/api'

interface ReviewSession {
  session_id: string
  status: string
  run_id: string
  created_at: string
  expires_at: string
  review_data: any
  content_for_review: {
    title?: string
    subtitle?: string
    abstract?: string
    content?: string
    word_count?: number
    reading_time_minutes?: number
    quality_scores?: any
    meta_description?: string
    keywords?: string[]
    tags?: string[]
  }
  has_feedback: boolean
}

interface ReviewFeedback {
  decision: 'approved' | 'rejected' | 'needs_revision'
  feedback_notes: string
  overall_rating?: number
  quality_concerns: string[]
}

export default function ReviewPage() {
  const params = useParams()
  const router = useRouter()
  const queryClient = useQueryClient()
  const sessionId = params.sessionId as string
  
  const [feedback, setFeedback] = useState<ReviewFeedback>({
    decision: 'approved',
    feedback_notes: '',
    overall_rating: 5,
    quality_concerns: []
  })
  
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null)

  // Fetch review session
  const { data: session, isLoading, error } = useQuery<ReviewSession>({
    queryKey: ['review-session', sessionId],
    queryFn: async () => {
      const response = await api.get(`/review/session/${sessionId}`)
      return response.data
    },
    enabled: !!sessionId,
    refetchInterval: 5000, // Poll every 5 seconds for updates
  })

  // Submit review feedback
  const submitReview = useMutation({
    mutationFn: async (reviewData: ReviewFeedback) => {
      const endpoint = reviewData.decision === 'approved' 
        ? `/review/session/${sessionId}/approve`
        : reviewData.decision === 'rejected'
        ? `/review/session/${sessionId}/reject`
        : `/review/session/${sessionId}/revise`
      
      const response = await api.post(endpoint, reviewData)
      return response.data
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['review-session', sessionId] })
      // Redirect to pipeline details page
      if (data.run_id) {
        router.push(`/pipelines/${data.run_id}`)
      }
    },
  })

  // Calculate time remaining
  useEffect(() => {
    if (session?.expires_at) {
      const updateTimeRemaining = () => {
        const now = new Date().getTime()
        const expires = new Date(session.expires_at).getTime()
        const remaining = Math.max(0, expires - now)
        setTimeRemaining(remaining)
      }
      
      updateTimeRemaining()
      const interval = setInterval(updateTimeRemaining, 1000)
      
      return () => clearInterval(interval)
    }
  }, [session?.expires_at])

  const formatTimeRemaining = (ms: number) => {
    const minutes = Math.floor(ms / 60000)
    const seconds = Math.floor((ms % 60000) / 1000)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const handleSubmit = (decision: 'approved' | 'rejected' | 'needs_revision') => {
    const reviewData = { ...feedback, decision }
    submitReview.mutate(reviewData)
  }

  const addQualityConcern = (concern: string) => {
    if (concern && !feedback.quality_concerns.includes(concern)) {
      setFeedback(prev => ({
        ...prev,
        quality_concerns: [...prev.quality_concerns, concern]
      }))
    }
  }

  const removeQualityConcern = (concern: string) => {
    setFeedback(prev => ({
      ...prev,
      quality_concerns: prev.quality_concerns.filter(c => c !== concern)
    }))
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
            <div className="card">
              <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
              <div className="space-y-3">
                <div className="h-4 bg-gray-200 rounded w-full"></div>
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-96 bg-gray-200 rounded"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !session) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-12">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Review Session Not Found</h1>
            <p className="text-secondary-600 mb-6">
              {error instanceof Error ? error.message : 'This review session may have expired or been completed.'}
            </p>
            <button
              onClick={() => router.push('/')}
              className="btn-primary"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (session.has_feedback) {
    return (
      <div className="min-h-screen bg-secondary-50">
        <Header />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-12">
            <h1 className="text-2xl font-bold text-green-600 mb-4">Review Complete</h1>
            <p className="text-secondary-600 mb-6">
              Feedback has already been submitted for this review session.
            </p>
            <button
              onClick={() => router.push(`/pipelines/${session.run_id}`)}
              className="btn-primary"
            >
              View Pipeline Details
            </button>
          </div>
        </div>
      </div>
    )
  }

  const content = session.content_for_review

  return (
    <div className="min-h-screen bg-secondary-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-3xl font-bold text-secondary-900">Content Review</h1>
              <p className="text-secondary-600 mt-2">
                Review and approve content before publishing
              </p>
            </div>
            
            {/* Time Remaining */}
            {timeRemaining !== null && timeRemaining > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="text-sm font-medium text-yellow-800">Time Remaining</div>
                <div className="text-2xl font-bold text-yellow-900">
                  {formatTimeRemaining(timeRemaining)}
                </div>
                <div className="text-xs text-yellow-600 mt-1">
                  Auto-approve on timeout
                </div>
              </div>
            )}
          </div>
          
          {/* Session Info */}
          <div className="flex items-center space-x-6 text-sm text-secondary-500">
            <span>Session: {sessionId.slice(0, 8)}</span>
            <span>Pipeline: {session.run_id.slice(0, 8)}</span>
            <span>Status: {session.status}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Content Preview */}
          <div className="lg:col-span-2">
            <div className="card">
              <div className="border-b border-secondary-200 pb-6 mb-6">
                <h2 className="text-2xl font-bold text-secondary-900 mb-2">
                  {content.title || 'Untitled Content'}
                </h2>
                {content.subtitle && (
                  <p className="text-lg text-secondary-600 mb-3">{content.subtitle}</p>
                )}
                
                {/* Content Metrics */}
                <div className="flex items-center space-x-6 text-sm text-secondary-500 mb-4">
                  {content.word_count && <span>{content.word_count} words</span>}
                  {content.reading_time_minutes && <span>{content.reading_time_minutes} min read</span>}
                </div>

                {/* Meta Description */}
                {content.meta_description && (
                  <div className="bg-secondary-50 rounded-lg p-4 mb-4">
                    <h4 className="font-medium text-secondary-900 mb-2">Meta Description</h4>
                    <p className="text-secondary-600 text-sm">{content.meta_description}</p>
                  </div>
                )}

                {/* Keywords & Tags */}
                {(content.keywords?.length || content.tags?.length) && (
                  <div className="space-y-3">
                    {content.keywords?.length && (
                      <div>
                        <h4 className="font-medium text-secondary-900 mb-2">Keywords</h4>
                        <div className="flex flex-wrap gap-2">
                          {content.keywords.map((keyword, idx) => (
                            <span key={idx} className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {content.tags?.length && (
                      <div>
                        <h4 className="font-medium text-secondary-900 mb-2">Tags</h4>
                        <div className="flex flex-wrap gap-2">
                          {content.tags.map((tag, idx) => (
                            <span key={idx} className="px-3 py-1 bg-secondary-100 text-secondary-700 rounded-full text-sm">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Abstract */}
              {content.abstract && (
                <div className="mb-6">
                  <h3 className="font-semibold text-secondary-900 mb-3">Abstract</h3>
                  <p className="text-secondary-600 leading-relaxed">{content.abstract}</p>
                </div>
              )}

              {/* Full Content */}
              <div>
                <h3 className="font-semibold text-secondary-900 mb-3">Content</h3>
                <div className="prose max-w-none">
                  <div className="bg-white border border-secondary-200 rounded-lg p-6 max-h-96 overflow-y-auto">
                    <pre className="whitespace-pre-wrap font-sans text-secondary-700">
                      {content.content || 'No content available'}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Review Panel */}
          <div className="space-y-6">
            {/* Quality Scores */}
            {content.quality_scores && (
              <div className="card">
                <h3 className="font-semibold text-secondary-900 mb-4">Quality Assessment</h3>
                <div className="space-y-3">
                  {Object.entries(content.quality_scores).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-sm text-secondary-600 capitalize">
                        {key.replace('_', ' ')}
                      </span>
                      <span className={`text-sm font-medium ${
                        (value as number) >= 0.8 ? 'text-green-600' : 
                        (value as number) >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {Math.round((value as number) * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Review Form */}
            <div className="card">
              <h3 className="font-semibold text-secondary-900 mb-4">Your Review</h3>
              
              {/* Overall Rating */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Overall Rating
                </label>
                <div className="flex space-x-2">
                  {[1, 2, 3, 4, 5].map(rating => (
                    <button
                      key={rating}
                      onClick={() => setFeedback(prev => ({ ...prev, overall_rating: rating }))}
                      className={`w-8 h-8 rounded-full text-sm font-medium transition-colors ${
                        feedback.overall_rating === rating
                          ? 'bg-primary-500 text-white'
                          : 'bg-secondary-100 text-secondary-600 hover:bg-primary-100'
                      }`}
                    >
                      {rating}
                    </button>
                  ))}
                </div>
              </div>

              {/* Feedback Notes */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Feedback Notes
                </label>
                <textarea
                  value={feedback.feedback_notes}
                  onChange={(e) => setFeedback(prev => ({ ...prev, feedback_notes: e.target.value }))}
                  placeholder="Add your comments, suggestions, or concerns..."
                  className="w-full px-3 py-2 border border-secondary-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  rows={4}
                />
              </div>

              {/* Quality Concerns */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Quality Concerns
                </label>
                <div className="space-y-2 mb-3">
                  {['Accuracy', 'Readability', 'SEO', 'Structure', 'Citations', 'Grammar'].map(concern => (
                    <label key={concern} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={feedback.quality_concerns.includes(concern)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            addQualityConcern(concern)
                          } else {
                            removeQualityConcern(concern)
                          }
                        }}
                        className="mr-2 h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                      />
                      <span className="text-sm text-secondary-700">{concern}</span>
                    </label>
                  ))}
                </div>
                
                {feedback.quality_concerns.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {feedback.quality_concerns.map(concern => (
                      <span 
                        key={concern}
                        className="px-2 py-1 bg-red-100 text-red-700 rounded text-sm cursor-pointer hover:bg-red-200"
                        onClick={() => removeQualityConcern(concern)}
                      >
                        {concern} ×
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={() => handleSubmit('approved')}
                  disabled={submitReview.isPending}
                  className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                >
                  {submitReview.isPending ? 'Submitting...' : 'Approve & Publish'}
                </button>
                
                <button
                  onClick={() => handleSubmit('needs_revision')}
                  disabled={submitReview.isPending}
                  className="w-full bg-yellow-600 text-white py-2 px-4 rounded-md hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-50"
                >
                  {submitReview.isPending ? 'Submitting...' : 'Request Revision'}
                </button>
                
                <button
                  onClick={() => handleSubmit('rejected')}
                  disabled={submitReview.isPending}
                  className="w-full bg-red-600 text-white py-2 px-4 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50"
                >
                  {submitReview.isPending ? 'Submitting...' : 'Reject Content'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}