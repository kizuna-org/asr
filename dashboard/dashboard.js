// Dashboard JavaScript for Rovo Dev CI/CD System

class Dashboard {
    constructor() {
        this.r2BaseUrl = 'https://your-r2-bucket.r2.cloudflarestorage.com'; // Configure this
        this.jobs = [];
        this.init();
    }

    init() {
        this.loadJobs();
        // Auto-refresh every 30 seconds
        setInterval(() => this.loadJobs(), 30000);
    }

    async loadJobs() {
        const loadingEl = document.getElementById('loading');
        const errorEl = document.getElementById('error');
        const jobsContainer = document.getElementById('jobs-container');

        try {
            loadingEl.style.display = 'block';
            errorEl.style.display = 'none';
            jobsContainer.style.display = 'none';

            // In a real implementation, you would fetch from R2 or a backend API
            // For now, we'll simulate with mock data
            const jobs = await this.fetchJobsFromR2();
            
            this.jobs = jobs;
            this.renderJobs();

            loadingEl.style.display = 'none';
            jobsContainer.style.display = 'grid';

        } catch (error) {
            console.error('Failed to load jobs:', error);
            loadingEl.style.display = 'none';
            errorEl.style.display = 'block';
        }
    }

    async fetchJobsFromR2() {
        // Mock data for demonstration
        // In production, this would fetch from R2 or a backend API that lists job statuses
        return [
            {
                jobId: 'a1b2c3d4e5f6',
                overallStatus: 'Succeeded',
                timestamps: {
                    created: '2025-06-18T14:30:00Z',
                    updated: '2025-06-18T14:55:00Z'
                },
                build: {
                    status: 'Succeeded',
                    log: '/a1b2c3d4e5f6/build.log',
                    imageUri: 'ghcr.io/user/repo:a1b2c3d4e5f6'
                },
                run: {
                    status: 'TaskSucceeded',
                    log: '/a1b2c3d4e5f6/app.log',
                    artifactUrl: 'https://huggingface.co/datasets/cicd-output-a1b2c3d4'
                }
            },
            {
                jobId: 'b2c3d4e5f6g7',
                overallStatus: 'Running',
                timestamps: {
                    created: '2025-06-18T15:00:00Z',
                    updated: '2025-06-18T15:10:00Z'
                },
                build: {
                    status: 'Succeeded',
                    log: '/b2c3d4e5f6g7/build.log',
                    imageUri: 'ghcr.io/user/repo:b2c3d4e5f6g7'
                },
                run: {
                    status: 'Running',
                    log: '/b2c3d4e5f6g7/app.log'
                }
            },
            {
                jobId: 'c3d4e5f6g7h8',
                overallStatus: 'BuildFailed',
                timestamps: {
                    created: '2025-06-18T14:00:00Z',
                    updated: '2025-06-18T14:05:00Z'
                },
                build: {
                    status: 'Failed',
                    log: '/c3d4e5f6g7h8/build.log'
                }
            }
        ];
    }

    renderJobs() {
        const container = document.getElementById('jobs-container');
        container.innerHTML = '';

        if (this.jobs.length === 0) {
            container.innerHTML = '<p class="loading">📭 ジョブが見つかりませんでした</p>';
            return;
        }

        // Sort jobs by creation time (newest first)
        const sortedJobs = this.jobs.sort((a, b) => 
            new Date(b.timestamps.created) - new Date(a.timestamps.created)
        );

        sortedJobs.forEach(job => {
            const jobCard = this.createJobCard(job);
            container.appendChild(jobCard);
        });
    }

    createJobCard(job) {
        const card = document.createElement('div');
        card.className = 'job-card';

        const statusClass = this.getStatusClass(job.overallStatus);
        const statusText = this.getStatusText(job.overallStatus);
        
        card.innerHTML = `
            <div class="job-header">
                <div class="job-id">Job ID: ${job.jobId}</div>
                <div class="status-badge ${statusClass}">${statusText}</div>
            </div>
            
            <div class="job-details">
                <div class="detail-row">
                    <span class="detail-label">作成日時:</span>
                    <span class="detail-value">${this.formatDate(job.timestamps.created)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">更新日時:</span>
                    <span class="detail-value">${this.formatDate(job.timestamps.updated)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">ビルド:</span>
                    <span class="detail-value">${job.build ? job.build.status : 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">実行:</span>
                    <span class="detail-value">${job.run ? job.run.status : 'N/A'}</span>
                </div>
            </div>
            
            <div class="actions">
                ${job.build && job.build.log ? `<a href="${this.r2BaseUrl}${job.build.log}" class="btn btn-secondary" target="_blank">ビルドログ</a>` : ''}
                ${job.run && job.run.log ? `<a href="${this.r2BaseUrl}${job.run.log}" class="btn btn-secondary" target="_blank">実行ログ</a>` : ''}
                ${job.run && job.run.artifactUrl ? `<a href="${job.run.artifactUrl}" class="btn btn-primary" target="_blank">成果物</a>` : ''}
            </div>
        `;

        return card;
    }

    getStatusClass(status) {
        const statusMap = {
            'Building': 'status-building',
            'BuildSucceeded': 'status-building',
            'Running': 'status-running',
            'Succeeded': 'status-succeeded',
            'BuildFailed': 'status-failed',
            'Failed': 'status-failed'
        };
        return statusMap[status] || 'status-building';
    }

    getStatusText(status) {
        const statusMap = {
            'Building': 'ビルド中',
            'BuildSucceeded': 'ビルド完了',
            'Running': '実行中',
            'Succeeded': '成功',
            'BuildFailed': 'ビルド失敗',
            'Failed': '失敗'
        };
        return statusMap[status] || status;
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('ja-JP', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// Global function for refresh button
function loadJobs() {
    if (window.dashboard) {
        window.dashboard.loadJobs();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});