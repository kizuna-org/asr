# Jenkins Gitea Job DSL修正記録

## 日付
2025-06-27

## 問題
JenkinsのConfiguration as Code (CasC) でJob DSLを使用してGiteaのMultibranch Pipelineジョブを作成しようとした際、以下のエラーが発生してJenkinsの起動に失敗：

```
groovy.lang.MissingMethodException: No signature of method: javaposse.jobdsl.dsl.helpers.workflow.BranchSourcesContext.gitea() is applicable for argument types...
Possible solutions: git(groovy.lang.Closure), gitlab(groovy.lang.Closure), grep(), github(groovy.lang.Closure)
```

## 原因
- Job DSLプラグインの現在のバージョンでは、`gitea()`メソッドが利用できない
- Giteaプラグインがインストールされていても、Job DSLでの直接サポートは提供されていない

## 解決策
Job DSLで汎用的な`git`ブランチソースを使用してGiteaリポジトリにアクセスする形式に変更：

### 変更前
```groovy
branchSources {
  gitea {
    id('gitea-source-from-jcasc')
    serverUrl("${GITEA_API_URL}")
    credentialsId("${GITEA_CREDENTIALS_ID}")
    repoOwner("${GITEA_USER}")
    repository("${GITEA_REPO}")
    traits {
      giteaBranchDiscovery {
        strategyId(1)
      }
    }
  }
}
```

### 変更後
```groovy
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
```

## 追加設定
- `configure`ブロックを追加してGiteaのwebhookサポートを設定
- `factory`ブロックを追加してJenkinsfileのパスを明示的に指定

## ファイル変更
- `poc/jenkins/casc.yaml`: Job DSLスクリプトを修正

## 検証予定
- Dockerコンテナの再起動
- Jenkins起動の成功確認
- Multibranch Pipelineジョブの作成確認

## 注意点
- この方法では、Gitea特有の機能（Pull Request検出など）は制限される可能性がある
- 基本的なブランチ検出とCI/CDパイプラインの実行は可能 
