'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { startPipeline } from '@/lib/api'

interface StartPipelineFormProps {
  onClose: () => void
  onStart: () => void
}

interface PipelineFormData {
  research_query: string
  domain_focus: string[]
  quality_thresholds: {
    overall: number
    technical: number
    domain_expertise: number
    style_consistency: number
    compliance: number
  }
}

export function StartPipelineForm({ onClose, onStart }: StartPipelineFormProps) {
  const queryClient = useQueryClient()
  const [selectedDomains, setSelectedDomains] = useState<string[]>(['ai'])
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset
  } = useForm<PipelineFormData>({
    defaultValues: {
      research_query: '',
      domain_focus: ['ai'],
      quality_thresholds: {
        overall: 0.85,
        technical: 0.90,
        domain_expertise: 0.90,
        style_consistency: 0.85,
        compliance: 0.95
      }
    }
  })

  const startPipelineMutation = useMutation({
    mutationFn: startPipeline,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipeline-stats'] })
      queryClient.invalidateQueries({ queryKey: ['recent-runs'] })
      reset()
      onStart()
    },
  })

  const domains = [
    { id: 'it_insurance', label: 'IT & Insurance', description: 'Cybersecurity, insurtech, digital transformation' },
    { id: 'ai', label: 'AI & Machine Learning', description: 'AI trends, applications, and developments' },
    { id: 'agentic_ai', label: 'Agentic AI', description: 'Multi-agent systems, LangGraph, ReAct patterns' },
    { id: 'ai_software_engineering', label: 'AI Software Engineering', description: 'AI in development, code generation' }
  ]

  const onSubmit = (data: PipelineFormData) => {
    const pipelineData = {
      ...data,
      domain_focus: selectedDomains,
      tenant_id: 'personal' // Phase 1: always personal
    }
    
    startPipelineMutation.mutate(pipelineData)
  }

  const toggleDomain = (domainId: string) => {
    if (selectedDomains.includes(domainId)) {
      setSelectedDomains(selectedDomains.filter(id => id !== domainId))
    } else {
      setSelectedDomains([...selectedDomains, domainId])
    }
  }

  return (
    <div className="border-t border-secondary-200 pt-6">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Research Query */}
        <div>
          <label className="block text-sm font-medium text-secondary-700 mb-2">
            Research Topic or Query
          </label>
          <textarea
            {...register('research_query', { 
              required: 'Please provide a research topic or query' 
            })}
            className="input-field h-20"
            placeholder="e.g., Latest developments in AI agents and multi-agent systems, or Current trends in cyber insurance..."
            disabled={startPipelineMutation.isPending}
          />
          {errors.research_query && (
            <p className="text-red-600 text-sm mt-1">{errors.research_query.message}</p>
          )}
        </div>

        {/* Domain Focus */}
        <div>
          <label className="block text-sm font-medium text-secondary-700 mb-3">
            Domain Focus
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {domains.map((domain) => (
              <div key={domain.id}>
                <label className="flex items-start space-x-3 p-3 border border-secondary-200 rounded-lg cursor-pointer hover:border-primary-300 transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedDomains.includes(domain.id)}
                    onChange={() => toggleDomain(domain.id)}
                    className="mt-1 rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                    disabled={startPipelineMutation.isPending}
                  />
                  <div>
                    <div className="font-medium text-secondary-900 text-sm">
                      {domain.label}
                    </div>
                    <div className="text-secondary-600 text-xs mt-1">
                      {domain.description}
                    </div>
                  </div>
                </label>
              </div>
            ))}
          </div>
          {selectedDomains.length === 0 && (
            <p className="text-red-600 text-sm mt-1">Please select at least one domain</p>
          )}
        </div>

        {/* Quality Thresholds */}
        <div>
          <label className="block text-sm font-medium text-secondary-700 mb-3">
            Quality Thresholds
          </label>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-secondary-600 mb-1">Overall</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                {...register('quality_thresholds.overall')}
                className="input-field text-sm"
                disabled={startPipelineMutation.isPending}
              />
            </div>
            <div>
              <label className="block text-xs text-secondary-600 mb-1">Technical</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                {...register('quality_thresholds.technical')}
                className="input-field text-sm"
                disabled={startPipelineMutation.isPending}
              />
            </div>
            <div>
              <label className="block text-xs text-secondary-600 mb-1">Domain Expertise</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                {...register('quality_thresholds.domain_expertise')}
                className="input-field text-sm"
                disabled={startPipelineMutation.isPending}
              />
            </div>
            <div>
              <label className="block text-xs text-secondary-600 mb-1">Style</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                {...register('quality_thresholds.style_consistency')}
                className="input-field text-sm"
                disabled={startPipelineMutation.isPending}
              />
            </div>
            <div>
              <label className="block text-xs text-secondary-600 mb-1">Compliance</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                {...register('quality_thresholds.compliance')}
                className="input-field text-sm"
                disabled={startPipelineMutation.isPending}
              />
            </div>
          </div>
        </div>

        {/* Error Display */}
        {startPipelineMutation.isError && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600 text-sm">
              Failed to start pipeline: {(startPipelineMutation.error as Error)?.message || 'Unknown error'}
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end space-x-3">
          <button
            type="button"
            onClick={onClose}
            className="btn-secondary"
            disabled={startPipelineMutation.isPending}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={startPipelineMutation.isPending || selectedDomains.length === 0}
          >
            {startPipelineMutation.isPending ? 'Starting...' : 'Start Pipeline'}
          </button>
        </div>
      </form>
    </div>
  )
}