// Jenkins Job DSL Examples for Gitea Integration
// このファイルは参考例として提供されています

// =================
// Multibranch Pipeline Job for Gitea
// =================
multibranchPipelineJob('gitea-multibranch-pipeline') {
    displayName('Gitea Multibranch Pipeline')
    description('Multibranch pipeline job that builds all branches from Gitea repository')
    
    branchSources {
        git {
            id('gitea-git-source')
            remote("${GITEA_API_URL}/${GITEA_USER}/${GITEA_REPO}.git")
            credentialsId("${GITEA_CREDENTIALS_ID}")
            traits {
                gitBranchDiscovery()
                // Pull Request検出を有効にする場合（Giteaでサポートされている場合）
                // gitTagDiscovery()
            }
        }
    }
    
    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile') // リポジトリ内のJenkinsfileパス
        }
    }
    
    // 自動スキャンの設定
    triggers {
        periodic(5) // 5分間隔でブランチをスキャン
    }
    
    // 不要な古いブランチの削除設定
    orphanedItemStrategy {
        discardOldItems {
            daysToKeep(7)
            numToKeep(20)
        }
    }
    
    // Gitea Webhook設定（configureブロックで直接XML設定）
    configure { project ->
        // Build Configuration
        project / 'factory' / 'scriptPath' << 'Jenkinsfile'
        
        // Additional properties can be set here
        project / 'properties' << 'jenkins.branch.BranchPropertyStrategy' {
            'properties' {
                'jenkins.branch.NoTriggerBranchProperty'()
            }
        }
    }
}

// =================
// Simple Pipeline Job for Gitea
// =================
pipelineJob('gitea-simple-pipeline') {
    displayName('Gitea Simple Pipeline')
    description('Simple pipeline job for a specific branch')
    
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url("${GITEA_API_URL}/${GITEA_USER}/${GITEA_REPO}.git")
                        credentials("${GITEA_CREDENTIALS_ID}")
                    }
                    branch('*/main') // メインブランチのみをビルド
                }
            }
            scriptPath('Jenkinsfile')
        }
    }
    
    // ビルドトリガーの設定
    triggers {
        scm('H/15 * * * *') // 15分間隔でSCMをポーリング
    }
    
    // パラメータの設定例
    parameters {
        stringParam('BRANCH_NAME', 'main', 'Branch to build')
        choiceParam('ENVIRONMENT', ['dev', 'staging', 'prod'], 'Deployment environment')
        booleanParam('RUN_TESTS', true, 'Run unit tests')
        booleanParam('DEPLOY', false, 'Deploy after successful build')
    }
    
    // ビルド履歴の保持設定
    logRotator {
        numToKeep(10)
        daysToKeep(30)
    }
}

// =================
// Freestyle Job Example
// =================
job('gitea-freestyle-job') {
    displayName('Gitea Freestyle Job')
    description('Freestyle job example for traditional build scripts')
    
    scm {
        git {
            remote {
                url("${GITEA_API_URL}/${GITEA_USER}/${GITEA_REPO}.git")
                credentials("${GITEA_CREDENTIALS_ID}")
            }
            branch('*/main')
        }
    }
    
    triggers {
        scm('H/10 * * * *')
    }
    
    steps {
        shell('''
            echo "Building project..."
            # ここにビルドコマンドを記述
            # make build
            # npm install && npm run build
            # go build
        ''')
    }
    
    publishers {
        archiveArtifacts {
            pattern('build/**/*')
            allowEmpty(false)
        }
        
        publishHtml([
            allowMissing: false,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'reports',
            reportFiles: 'index.html',
            reportName: 'Build Report'
        ])
    }
}

// =================
// Folder Structure for Organization
// =================
folder('gitea-projects') {
    displayName('Gitea Projects')
    description('Folder containing all Gitea-related Jenkins jobs')
}

// フォルダ内にジョブを作成
multibranchPipelineJob('gitea-projects/my-app-pipeline') {
    displayName('My Application Pipeline')
    description('Pipeline for my application')
    
    branchSources {
        git {
            id('my-app-source')
            remote("${GITEA_API_URL}/${GITEA_USER}/my-app.git")
            credentialsId("${GITEA_CREDENTIALS_ID}")
            traits {
                gitBranchDiscovery()
            }
        }
    }
    
    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile')
        }
    }
} 
