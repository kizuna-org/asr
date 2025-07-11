# 2025-06-29 Startup Trigger Implementation

## 概要
Jenkins起動時にmy-project-seed-jobが確実に実行されるようにstartup trigger機能を実装しました。

## 実装したアプローチ

### 最初の試行: startup-trigger-plugin
- `startup-trigger-plugin`をplugins.txtに追加
- Job DSLで`startup()`メソッドを使用する設定を試行
- 結果: Job DSLで`startup()`メソッドがサポートされていないため失敗

### 最終実装: Groovy Init Script
Jenkins起動時に実行されるGroovyスクリプトを使用した方法で実装しました。

#### 1. Dockerfileの修正
```dockerfile
# Create init.groovy.d directory and add startup script
RUN mkdir -p /usr/share/jenkins/ref/init.groovy.d
COPY startup-trigger.groovy /usr/share/jenkins/ref/init.groovy.d/startup-trigger.groovy
```

#### 2. startup-trigger.groovyの作成
```groovy
import jenkins.model.Jenkins
import hudson.model.FreeStyleProject
import hudson.model.Cause

// Jenkins起動時にseed jobを実行
println "Running startup trigger script..."

def jenkins = Jenkins.getInstance()

// Jenkins が完全に起動するまで少し待機
Thread.sleep(10000)

try {
    def job = jenkins.getItemByFullName('my-project-seed-job')
    if (job != null) {
        println "Found my-project-seed-job, triggering build..."
        job.scheduleBuild(0, new Cause.UserIdCause("startup-script"))
        println "Seed job triggered successfully on startup"
    } else {
        println "my-project-seed-job not found, it may not be created yet"
    }
} catch (Exception e) {
    println "Error triggering seed job: ${e.getMessage()}"
}
```

#### 3. plugins.txtの更新
不要なstartup-trigger-pluginを削除しました。

## 実行結果の確認

### Docker Composeでの確認
```bash
docker compose down && rm -rf jenkins_home && docker compose --progress plain build
docker compose --progress plain up -d
docker compose ps
```

### ログでの確認
```bash
docker compose logs jenkins | grep -E "(startup|groovy|seed|trigger)"
```

ログ出力:
```
jenkins-1  | 2025-06-29 11:19:08.176+0000 [id=53]       INFO    j.util.groovy.GroovyHookScript#execute: Executing /var/jenkins_home/init.groovy.d/startup-trigger.groovy
jenkins-1  | Running startup trigger script...
jenkins-1  | Found my-project-seed-job, triggering build...
jenkins-1  | Seed job triggered successfully on startup
```

## 技術的詳細

### Groovy Init Scriptの利点
1. **確実性**: Jenkins起動プロセスの一部として実行される
2. **シンプル**: 追加プラグインが不要
3. **デバッグ可能**: ログで実行状況を確認できる

### タイミング制御
- Jenkins完全起動まで10秒待機
- my-project-seed-jobの存在確認
- エラーハンドリング付きでジョブをスケジュール

## 成果
- ✅ Jenkins起動時にmy-project-seed-jobが自動実行される
- ✅ ログで実行確認が可能
- ✅ エラーハンドリングが実装されている
- ✅ 追加プラグインが不要

## 今後の改善点
- 待機時間の最適化（現在は固定10秒）
- より詳細なエラーログの追加
- ジョブ実行失敗時のリトライ機能 
