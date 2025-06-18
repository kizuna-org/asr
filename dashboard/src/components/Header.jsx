import React from 'react'
import { Activity, Clock } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { ja } from 'date-fns/locale'

const Header = ({ lastUpdated }) => {
  return (
    <header className="bg-gradient-to-r from-indigo-600 via-purple-600 to-blue-600 text-white shadow-2xl">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-white/20 rounded-full backdrop-blur-sm">
              <Activity className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                🚀 Rovo Dev CI/CD Dashboard
              </h1>
              <p className="text-blue-100 text-lg mt-1">
                リアルタイムパイプライン監視システム
              </p>
            </div>
          </div>
          
          {lastUpdated && (
            <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
              <Clock className="w-4 h-4" />
              <span className="text-sm">
                最終更新: {formatDistanceToNow(lastUpdated, { 
                  addSuffix: true, 
                  locale: ja 
                })}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header