// API service for fetching job data from Cloudflare R2
// This is a mock implementation - replace with actual R2 integration

import type { Job } from '../types'

const MOCK_JOBS: Job[] = [
  {
    jobId: "a1b2c3d4e5f6",
    overallStatus: "Succeeded",
    timestamps: {
      created: "2025-06-18T14:30:00Z",
      updated: "2025-06-18T14:55:00Z"
    },
    build: {
      status: "Succeeded",
      log: "/a1b2c3d4e5f6/build.log",
      imageUri: "ghcr.io/user/repo:a1b2c3d4e5f6"
    },
    run: {
      status: "Succeeded",
      log: "/a1b2c3d4e5f6/app.log",
      artifactUrl: "https://huggingface.co/user/model/a1b2c3d4e5f6"
    }
  },
  {
    jobId: "f6e5d4c3b2a1",
    overallStatus: "Running",
    timestamps: {
      created: "2025-06-18T15:00:00Z",
      updated: "2025-06-18T15:10:00Z"
    },
    build: {
      status: "Succeeded",
      log: "/f6e5d4c3b2a1/build.log",
      imageUri: "ghcr.io/user/repo:f6e5d4c3b2a1"
    },
    run: {
      status: "Running",
      log: "/f6e5d4c3b2a1/app.log"
    }
  },
  {
    jobId: "1a2b3c4d5e6f",
    overallStatus: "Failed",
    timestamps: {
      created: "2025-06-18T13:45:00Z",
      updated: "2025-06-18T13:50:00Z"
    },
    build: {
      status: "Failed",
      log: "/1a2b3c4d5e6f/build.log"
    }
  },
  {
    jobId: "9z8y7x6w5v4u",
    overallStatus: "Building",
    timestamps: {
      created: "2025-06-18T15:20:00Z",
      updated: "2025-06-18T15:22:00Z"
    },
    build: {
      status: "Building",
      log: "/9z8y7x6w5v4u/build.log"
    }
  }
]

// Simulate API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

export const fetchJobs = async () => {
  // Simulate network delay
  await delay(Math.random() * 1000 + 500)
  
  // Simulate occasional errors for testing
  if (Math.random() < 0.1) {
    throw new Error('ネットワークエラー: サーバーに接続できませんでした')
  }
  
  // Sort by creation time (newest first)
  return MOCK_JOBS.sort((a, b) => 
    new Date(b.timestamps.created) - new Date(a.timestamps.created)
  )
}

// TODO: Replace with actual Cloudflare R2 integration
// Example implementation:
/*
export const fetchJobs = async () => {
  try {
    // Fetch list of job folders from R2
    const response = await fetch('/api/jobs')
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const jobs = await response.json()
    return jobs.sort((a, b) => 
      new Date(b.timestamps.created) - new Date(a.timestamps.created)
    )
  } catch (error) {
    console.error('Failed to fetch jobs:', error)
    throw new Error('ジョブデータの取得に失敗しました')
  }
}
*/