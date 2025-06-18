import React from 'react'
import { Loader2 } from 'lucide-react'

const LoadingSpinner = () => {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-slate-200 border-t-primary-600 rounded-full animate-spin"></div>
        <Loader2 className="w-8 h-8 text-primary-600 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-pulse" />
      </div>
      <h3 className="text-xl font-semibold text-slate-700 mt-6 mb-2">
        📊 ジョブデータを読み込み中...
      </h3>
      <p className="text-slate-500 text-center max-w-md">
        パイプラインの状況を取得しています。しばらくお待ちください。
      </p>
    </div>
  )
}

export default LoadingSpinner