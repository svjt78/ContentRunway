'use client'

import { useState } from 'react'
import { PipelineOverview } from '@/components/dashboard/PipelineOverview'
import { StartPipelineForm } from '@/components/pipeline/StartPipelineForm'
import { RecentRuns } from '@/components/dashboard/RecentRuns'
import { Header } from '@/components/layout/Header'

export default function HomePage() {
  const [showStartForm, setShowStartForm] = useState(false)

  return (
    <div className="min-h-screen bg-secondary-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-secondary-900 mb-2">
            ContentRunway Dashboard
          </h1>
          <p className="text-secondary-600">
            AI-powered content creation pipeline with quality-first approach
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Pipeline Overview & Controls */}
          <div className="lg:col-span-2 space-y-6">
            <PipelineOverview />
            
            {/* Start New Pipeline */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-secondary-900">
                  Content Pipeline
                </h2>
                <button
                  onClick={() => setShowStartForm(!showStartForm)}
                  className="btn-primary"
                >
                  Start New Pipeline
                </button>
              </div>
              
              {showStartForm && (
                <StartPipelineForm 
                  onClose={() => setShowStartForm(false)}
                  onStart={() => setShowStartForm(false)}
                />
              )}
            </div>
          </div>

          {/* Right Column - Recent Activity */}
          <div className="space-y-6">
            <RecentRuns />
          </div>
        </div>
      </main>
    </div>
  )
}