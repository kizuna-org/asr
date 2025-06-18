import React from 'react'
import { RefreshCw } from 'lucide-react'

interface RefreshButtonProps {
  onRefresh: () => void
  loading: boolean
}

const RefreshButton: React.FC<RefreshButtonProps> = ({ onRefresh, loading }) => {
  return (
    <button
      onClick={onRefresh}
      disabled={loading}
      className="fixed bottom-6 right-6 p-4 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 group"
      title="更新"
    >
      <RefreshCw 
        className={`w-6 h-6 transition-transform duration-300 ${
          loading ? 'animate-spin' : 'group-hover:rotate-180'
        }`} 
      />
    </button>
  )
}

export default RefreshButton