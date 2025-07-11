# Jenkins Web UI設定からJob DSL変換ガイド

## 日付
2025-06-27

## 概要
JenkinsのWeb UIで手動設定したパイプラインジョブをJob DSLスクリプトに変換し、Configuration as Code (CasC) で管理する方法を説明します。

## 手順

### 1. 既存ジョブの設定情報取得

#### 方法A: Jenkins REST APIを使用
```bash
# XMLフォーマットでジョブ設定を取得
curl -u "${JENKINS_USER}:${JENKINS_PASS}" \
  "http://localhost:8080/job/${JOB_NAME}/config.xml" \
  -o job_config.xml
```

#### 方法B: Jenkins Web UIから設定内容を記録
1. **Multibranch Pipeline**の場合：
   - Branch Sources（ブランチソース）の設定
   - Build Configuration（ビルド設定）
   - Scan Multibranch Pipeline Triggers（スキャントリガー）
   - Property strategy（プロパティ戦略）

2. **Pipeline Job**の場合：
   - Pipeline Script（パイプラインスクリプト）
   - Pipeline Script from SCM（SCMからのスクリプト）
   - Build Triggers（ビルドトリガー）

### 2. Job DSLスクリプトへの変換

#### Multibranch Pipeline の場合

```groovy
// Job DSL for Multibranch Pipeline
multibranchPipelineJob('my-multibranch-pipeline') {
    displayName('My Multibranch Pipeline')
    description('Converted from Web UI configuration')
    
    branchSources {
        git {
            id('git-source-id')
            remote('https://github.com/user/repository.git')
            credentialsId('git-credentials')
            traits {
                gitBranchDiscovery()
                gitTagDiscovery()
            }
        }
    }
    
    factory {
        workflowBranchProjectFactory {
            scriptPath('Jenkinsfile')
        }
    }
    
    triggers {
        periodic(1) // スキャン間隔（分）
    }
    
    configure { project ->
        // 追加のXML設定があればここで指定
        project / sources / data / 'jenkins.branch.BranchSource' / source / traits << {
            'jenkins.plugins.git.traits.CleanBeforeCheckoutTrait' {
                extension(class: 'hudson.plugins.git.extensions.impl.CleanCheckout')
            }
        }
    }
}
```

#### 通常のPipeline Jobの場合

```groovy
// Job DSL for Pipeline Job
pipelineJob('my-pipeline-job') {
    displayName('My Pipeline Job')
    description('Converted from Web UI configuration')
    
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('https://github.com/user/repository.git')
                        credentials('git-credentials')
                    }
                    branch('*/main')
                }
            }
            scriptPath('Jenkinsfile')
        }
    }
    
    triggers {
        scm('H/15 * * * *') // SCMポーリング
    }
    
    parameters {
        stringParam('BRANCH_NAME', 'main', 'Branch to build')
        booleanParam('DEPLOY', false, 'Deploy after build')
    }
}
```

### 3. CasC.yamlへの統合

Job DSLスクリプトを`casc.yaml`に統合：

```yaml
jobs:
  - script: |
      multibranchPipelineJob('gitea-multibranch-pipeline') {
        displayName('Gitea Multibranch Pipeline')
        description('Pipeline job for Gitea repository')
        
        branchSources {
          git {
            id('gitea-git-source')
            remote("${GITEA_API_URL}/${GITEA_USER}/${GITEA_REPO}.git")
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
        
        triggers {
          periodic(5)
        }
        
        configure { project ->
          // Gitea webhook設定
          project / triggers << 'com.cloudbees.jenkins.GitHubPushTrigger' {
            spec('')
          }
        }
      }
```

### 4. 変換時の注意点

#### よくある設定とJob DSL対応

| Web UI設定 | Job DSL記述 |
|------------|-------------|
| Repository URL | `remote('url')` |
| Credentials | `credentialsId('id')` |
| Branch Specifier | `branch('pattern')` |
| Poll SCM | `scm('cron-expression')` |
| Build periodically | `cron('cron-expression')` |
| GitHub hook trigger | `githubPush()` |

#### プラグイン固有の設定

```groovy
// Gitea Webhook（configureブロックで設定）
configure { project ->
    project / 'properties' / 'org.jenkinsci.plugins.workflow.multibranch.BranchJobProperty' / 'strategy' {
        'jenkins.branch.DefaultBranchPropertyStrategy' {
            'properties' {
                'jenkins.branch.NoTriggerBranchProperty'()
            }
        }
    }
}
```

### 5. 検証手順

1. **Dry Run実行**
```bash
# Jenkins起動前にJob DSL構文チェック
docker-compose up --dry-run
```

2. **Jenkins起動とジョブ確認**
```bash
# Jenkinsコンテナ起動
docker-compose up -d jenkins

# ジョブが正常に作成されたか確認
curl -u admin:password http://localhost:8080/api/json
```

3. **設定比較**
   - 元のWeb UI設定と生成されたジョブ設定を比較
   - 不足している設定があれば Job DSL に追加

### 6. トラブルシューティング

#### よくあるエラーと対処法

- **`MissingMethodException`**: プラグインが対応していないメソッド
  → `configure`ブロックでXML直接指定
  
- **認証エラー**: Credentials IDの不一致
  → `credentials`セクションで事前に定義

- **Branch Discovery設定エラー**: 
  → `traits`ブロックで適切なtraitsを指定

## 参考資料

- [Job DSL Plugin Wiki](https://github.com/jenkinsci/job-dsl-plugin/wiki)
- [Job DSL API Reference](https://jenkinsci.github.io/job-dsl-plugin/)
- [Configuration as Code Plugin](https://github.com/jenkinsci/configuration-as-code-plugin) 
