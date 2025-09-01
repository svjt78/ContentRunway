import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token')
      // Redirect to login if needed
    }
    return Promise.reject(error)
  }
)

// Pipeline API
export interface StartPipelineRequest {
  research_query: string
  domain_focus: string[]
  quality_thresholds: {
    overall: number
    technical: number
    domain_expertise: number
    style_consistency: number
    compliance: number
  }
  tenant_id: string
}

export interface PipelineRun {
  id: string
  tenant_id: string
  status: string
  domain_focus: string[]
  quality_thresholds: Record<string, number>
  created_at: string
  started_at?: string
  completed_at?: string
  current_step: string
  progress_percentage: number
  chosen_topic_id?: string
  final_quality_score?: number
  human_approved: boolean
  published_urls?: string[]
  error_message?: string
}

export interface PipelineStats {
  total_runs: number
  active_runs: number
  completed_runs: number
  failed_runs: number
  success_rate: number
  avg_processing_time: number
}

export const startPipeline = async (data: StartPipelineRequest): Promise<PipelineRun> => {
  const response = await api.post('/pipeline/start', data)
  return response.data
}

export const getPipelineRun = async (runId: string): Promise<PipelineRun> => {
  const response = await api.get(`/pipeline/runs/${runId}`)
  return response.data
}

export const getPipelineRuns = async (params?: {
  limit?: number
  offset?: number
  status?: string
}): Promise<PipelineRun[]> => {
  const response = await api.get('/pipeline/runs', { params })
  return response.data
}

export const getRecentPipelineRuns = async (limit: number = 10): Promise<PipelineRun[]> => {
  const response = await api.get('/pipeline/runs', { 
    params: { limit, offset: 0 } 
  })
  return response.data
}

export const getPipelineStats = async (): Promise<PipelineStats> => {
  try {
    // This would be a real API endpoint
    const response = await api.get('/pipeline/stats')
    return response.data
  } catch (error) {
    // Return default stats if API not available
    return {
      total_runs: 0,
      active_runs: 0,
      completed_runs: 0,
      failed_runs: 0,
      success_rate: 0,
      avg_processing_time: 0
    }
  }
}

export const pausePipeline = async (runId: string): Promise<void> => {
  await api.post(`/pipeline/runs/${runId}/pause`)
}

export const resumePipeline = async (runId: string): Promise<void> => {
  await api.post(`/pipeline/runs/${runId}/resume`)
}

export const cancelPipeline = async (runId: string): Promise<void> => {
  await api.delete(`/pipeline/runs/${runId}`)
}

// Content API
export interface TopicIdea {
  id: string
  title: string
  description: string
  domain: string
  relevance_score: number
  novelty_score: number
  seo_difficulty: number
  overall_score: number
  target_keywords: string[]
  is_selected: boolean
}

export const getPipelineTopics = async (runId: string): Promise<TopicIdea[]> => {
  const response = await api.get(`/pipeline/runs/${runId}/topics`)
  return response.data
}

export const selectTopic = async (runId: string, topicId: string): Promise<void> => {
  await api.post(`/pipeline/runs/${runId}/topics/${topicId}/select`)
}

// Quality API
export interface QualityAssessment {
  id: string
  gate_name: string
  overall_score: number
  passed: boolean
  criteria_scores: Record<string, number>
  strengths: string[]
  weaknesses: string[]
  suggestions: string[]
}

export const getQualityAssessments = async (runId: string): Promise<QualityAssessment[]> => {
  const response = await api.get(`/quality/assessments/${runId}`)
  return response.data
}

// Review API
export interface HumanReview {
  id: string
  status: string
  decision?: string
  overall_rating?: number
  feedback_notes?: string
  time_spent_seconds?: number
}

export const getHumanReview = async (runId: string): Promise<HumanReview> => {
  const response = await api.get(`/review/${runId}`)
  return response.data
}

export const submitReview = async (
  runId: string, 
  reviewData: {
    decision: string
    overall_rating: number
    feedback_notes: string
    inline_edits?: any[]
  }
): Promise<void> => {
  await api.post(`/review/${runId}/submit`, reviewData)
}

export default api