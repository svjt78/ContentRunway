'use client'

import Link from 'next/link'

export function Header() {
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