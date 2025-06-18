import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import JobGrid from './components/JobGrid'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorMessage from './components/ErrorMessage'
import RefreshButton from './components/RefreshButton'
import { fetchJobs } from './services/api'

function App() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const loadJobs = async () => {
    try {
      setLoading(true)
      setError(null)
      const jobsData = await fetchJobs()
      setJobs(jobsData)
      setLastUpdated(new Date())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadJobs()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadJobs, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <Header lastUpdated={lastUpdated} />
      
      <main className="container mx-auto px-4 py-8">
        {loading && jobs.length === 0 ? (
          <LoadingSpinner />
        ) : error ? (
          <ErrorMessage message={error} onRetry={loadJobs} />
        ) : (
          <JobGrid jobs={jobs} loading={loading} />
        )}
      </main>
      
      <RefreshButton onRefresh={loadJobs} loading={loading} />
    </div>
  )
}

export default App