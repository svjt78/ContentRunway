'use client'

import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export function Header() {
  // Check for pending reviews using new content-based system
  const { data: pendingContent } = useQuery({
    queryKey: ['pending-content-reviews'],
    queryFn: async () => {
      try {
        const response = await api.get('/content/pending-review')
        return response.data || []
      } catch (error) {
        console.error('Failed to fetch pending content:', error)
        return []
      }
    },
    refetchInterval: 30000, // Check every 30 seconds
  })
  return (
    <header className="bg-white shadow-sm border-b border-secondary-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">CR</span>
              </div>
              <span className="ml-2 text-xl font-bold text-secondary-900">
                ContentRunway
              </span>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex space-x-6">
            <Link 
              href="/" 
              className="text-secondary-600 hover:text-secondary-900 font-medium"
            >
              Dashboard
            </Link>
            <Link 
              href="/pipelines" 
              className="text-secondary-600 hover:text-secondary-900 font-medium"
            >
              Pipelines
            </Link>
            <Link 
              href="/content" 
              className="text-secondary-600 hover:text-secondary-900 font-medium"
            >
              Content
            </Link>
            <Link 
              href="/analytics" 
              className="text-secondary-600 hover:text-secondary-900 font-medium"
            >
              Analytics
            </Link>
          </nav>

          {/* User Menu */}
          <div className="flex items-center space-x-4">
            {/* Review Notification */}
            {pendingContent && pendingContent.length > 0 && (
              <Link 
                href="/content?filter=pending&highlight=true"
                className="relative bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm font-medium hover:bg-orange-200 transition-colors"
              >
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-orange-500 rounded-full mr-2 animate-pulse" />
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  {pendingContent.length} Review{pendingContent.length > 1 ? 's' : ''} Pending
                </div>
              </Link>
            )}
            
            <div className="text-sm text-secondary-600">
              Personal Account
            </div>
            <div className="w-8 h-8 bg-secondary-300 rounded-full flex items-center justify-center">
              <span className="text-secondary-700 font-medium text-sm">P</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}