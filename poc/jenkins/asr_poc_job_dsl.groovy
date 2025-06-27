// Job DSL for asr-poc Multibranch Pipeline
// Generated from Web UI configuration on 2025-06-27

multibranchPipelineJob('asr-poc') {
    displayName('asr-poc')
    description('Multibranch pipeline job for ASR POC project')
    
    // ジョブを有効にする
    disabled(false)
    
    // Branch Sources設定
    branchSources {
        git {
            id('asr-poc-git-source')
            // Giteaリポジトリの設定
            remote('http://gitea:3000/poc_user/my-poc-repo.git')
            credentialsId('poc_user')
            
            // Behaviours設定
            traits {
                gitBranchDiscovery() // Discover branches
            }
        }
    }
    
    // Build Configuration設定
    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile') // JenkinsfileのパスS
        }
    }
    
    // Scan Multibranch Pipeline Triggers設定
    triggers {
        // 定期的なスキャンは無効（チェックボックスがオフだったため）
    }
    
    // Properties設定
    properties {
        // Pipeline Libraries（現在は空）
    }
    
    // 不要アイテムの扱い
    orphanedItemStrategy {
        discardOldItems {
            // 古いアイテムを削除する設定
            // 保持日数と保持個数は設定されていなかった
        }
    }
    
    // Appearance設定
    configure { node ->
        // Metadata Folder Iconの設定
        node / 'icon' / 'hudson.plugins.folderplus.FolderPlusIconProperty' {
            iconClassName('icon-folder-plus-metadata')
        }
    }
} 
