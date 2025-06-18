import React from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="p-4 bg-error-100 rounded-full mb-6">
        <AlertTriangle className="w-12 h-12 text-error-600" />
      </div>
      
      <h3 className="text-xl font-semibold text-error-700 mb-2">
        ❌ データの読み込みに失敗しました
      </h3>
      
      <p className="text-error-600 text-center max-w-md mb-6">
        {message || 'ネットワークエラーまたはサーバーエラーが発生しました。'}
      </p>
      
      <button
        onClick={onRetry}
        className="inline-flex items-center space-x-2 px-6 py-3 bg-error-600 hover:bg-error-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-error-500 focus:ring-offset-2"
      >
        <RefreshCw className="w-4 h-4" />
        <span>再試行</span>
      </button>
    </div>
  )
}

export default ErrorMessage