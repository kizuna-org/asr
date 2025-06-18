// Type definitions based on schemas/log_schema.json

export interface LogEntry {
  timestamp: string; // ISO 8601 timestamp in UTC
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR' | 'FATAL';
  component: 'app' | 'build_subscriber' | 'app_subscriber' | 'github_actions';
  message: string;
  job_id?: string;
  context?: {
    operation?: string;
    step?: string;
    progress?: {
      current: number;
      total: number;
      percentage: number;
    };
    metadata?: Record<string, any>;
  };
  error?: {
    type?: string;
    code?: string;
    details?: string;
    stack_trace?: string;
  };
  performance?: {
    duration_ms?: number;
    memory_usage_mb?: number;
    cpu_usage_percent?: number;
  };
  tags?: string[];
}

export interface JobStatus {
  status: 'Succeeded' | 'Failed' | 'Running' | 'Building' | 'Pending';
  log?: string;
  imageUri?: string;
  artifactUrl?: string;
}

export interface Job {
  jobId: string;
  overallStatus: 'Succeeded' | 'Failed' | 'Running' | 'Building' | 'Pending';
  timestamps: {
    created: string;
    updated: string;
  };
  build?: JobStatus;
  run?: JobStatus;
}

export interface ApiResponse<T> {
  data: T;
  error?: string;
}

export interface DashboardProps {
  jobs: Job[];
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

export interface ComponentProps {
  job?: Job;
  jobs?: Job[];
  loading?: boolean;
  error?: string | null;
  message?: string;
  lastUpdated?: Date | null;
  onRefresh?: () => void;
  onRetry?: () => void;
}