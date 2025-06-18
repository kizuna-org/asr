import React from 'react'
import JobCard from './JobCard'
import { Package } from 'lucide-react'

const JobGrid = ({ jobs, loading }) => {
  if (jobs.length === 0 && !loading) {
    return (
      <div className="text-center py-16">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-slate-100 rounded-full mb-4">
          <Package className="w-8 h-8 text-slate-400" />
        </div>
        <h3 className="text-xl font-semibold text-slate-600 mb-2">
          ジョブが見つかりません
        </h3>
        <p className="text-slate-500">
          まだパイプラインが実行されていないか、データの読み込み中です。
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">
          パイプライン実行状況
        </h2>
        <div className="text-sm text-slate-600">
          {jobs.length} 件のジョブ
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {jobs.map((job) => (
          <JobCard key={job.jobId} job={job} />
        ))}
      </div>
      
      {loading && jobs.length > 0 && (
        <div className="text-center py-4">
          <div className="inline-flex items-center space-x-2 text-slate-600">
            <div className="w-4 h-4 border-2 border-slate-300 border-t-slate-600 rounded-full animate-spin"></div>
            <span>更新中...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default JobGrid