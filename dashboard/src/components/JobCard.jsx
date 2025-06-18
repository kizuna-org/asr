import React from 'react'
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Play, 
  Package,
  ExternalLink,
  FileText,
  GitCommit
} from 'lucide-react'
import { formatDistanceToNow, format } from 'date-fns'
import { ja } from 'date-fns/locale'

const JobCard = ({ job }) => {
  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'succeeded':
      case 'success':
        return <CheckCircle className="w-5 h-5 text-success-600" />
      case 'failed':
      case 'error':
        return <XCircle className="w-5 h-5 text-error-600" />
      case 'running':
      case 'building':
        return <Play className="w-5 h-5 text-warning-600 animate-pulse" />
      default:
        return <AlertCircle className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusBadge = (status) => {
    const baseClasses = "status-badge"
    switch (status?.toLowerCase()) {
      case 'succeeded':
      case 'success':
        return `${baseClasses} bg-success-100 text-success-700`
      case 'failed':
      case 'error':
        return `${baseClasses} bg-error-100 text-error-700`
      case 'running':
      case 'building':
        return `${baseClasses} bg-warning-100 text-warning-700`
      default:
        return `${baseClasses} bg-slate-100 text-slate-700`
    }
  }

  const formatJobId = (jobId) => {
    return jobId?.length > 8 ? `${jobId.substring(0, 8)}...` : jobId
  }

  const createdDate = job.timestamps?.created ? new Date(job.timestamps.created) : null
  const updatedDate = job.timestamps?.updated ? new Date(job.timestamps.updated) : null

  return (
    <div className="glass-effect rounded-xl p-6 card-hover animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-slate-100 rounded-lg">
            <GitCommit className="w-5 h-5 text-slate-600" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">
              Job {formatJobId(job.jobId)}
            </h3>
            <p className="text-sm text-slate-500 font-mono">
              {job.jobId}
            </p>
          </div>
        </div>
        <div className={getStatusBadge(job.overallStatus)}>
          {getStatusIcon(job.overallStatus)}
          <span className="ml-1">{job.overallStatus || 'Unknown'}</span>
        </div>
      </div>

      {/* Timestamps */}
      <div className="space-y-2 mb-4">
        {createdDate && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-500">作成:</span>
            <span className="text-slate-700">
              {formatDistanceToNow(createdDate, { addSuffix: true, locale: ja })}
            </span>
          </div>
        )}
        {updatedDate && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-500">更新:</span>
            <span className="text-slate-700">
              {formatDistanceToNow(updatedDate, { addSuffix: true, locale: ja })}
            </span>
          </div>
        )}
      </div>

      {/* Build Status */}
      {job.build && (
        <div className="border-t border-slate-200 pt-4 mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Package className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-medium text-slate-700">ビルド</span>
            </div>
            <div className={getStatusBadge(job.build.status)}>
              {getStatusIcon(job.build.status)}
              <span className="ml-1">{job.build.status}</span>
            </div>
          </div>
          {job.build.imageUri && (
            <p className="text-xs text-slate-500 font-mono truncate">
              {job.build.imageUri}
            </p>
          )}
        </div>
      )}

      {/* Run Status */}
      {job.run && (
        <div className="border-t border-slate-200 pt-4 mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Play className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-medium text-slate-700">実行</span>
            </div>
            <div className={getStatusBadge(job.run.status)}>
              {getStatusIcon(job.run.status)}
              <span className="ml-1">{job.run.status}</span>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap gap-2 pt-4 border-t border-slate-200">
        {job.build?.log && (
          <a
            href={job.build.log}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs rounded-lg transition-colors"
          >
            <FileText className="w-3 h-3" />
            <span>ビルドログ</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
        {job.run?.log && (
          <a
            href={job.run.log}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs rounded-lg transition-colors"
          >
            <FileText className="w-3 h-3" />
            <span>実行ログ</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
        {job.run?.artifactUrl && (
          <a
            href={job.run.artifactUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1 px-3 py-1.5 bg-primary-100 hover:bg-primary-200 text-primary-700 text-xs rounded-lg transition-colors"
          >
            <Package className="w-3 h-3" />
            <span>成果物</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>
    </div>
  )
}

export default JobCard