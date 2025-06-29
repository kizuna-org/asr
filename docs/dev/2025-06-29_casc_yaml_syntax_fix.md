# 2025-06-29: Jenkins Configuration as Code YAML構文エラー修正

## 概要
Jenkins Configuration as Code（JCasC）のYAMLファイル（`poc/jenkins/casc.yaml`）において、`jobs.script`セクションの構文エラーを特定・修正し、Seed Jobの自動実行設定も追加しました。

## 発生していた問題
- Jenkinsの起動時に以下のエラーが発生：
```
script: 24: unexpected token: } @ line 24, column 1.
   }
   ^
1 error
```

## 原因の特定プロセス
段階的なコメントアウトテストを実施：

1. **jobs.scriptセクション全体をコメントアウト** → 正常起動
2. **最小限のjobブロックのみ有効化** → 正常起動  
3. **SCMセクション追加** → 正常起動
4. **stepsセクション（jobDsl）追加** → 正常起動
5. **元の設定（日本語コメント含む）復元** → エラー発生

## 特定された根本原因
YAMLの`script: >`ブロック内で記述されるGroovyコードに以下の問題があった：

1. **日本語コメント**: 
   - `// この job {} ブロックはSeed Jobを定義します`
   - `// ジョブ定義(DSL)を管理するGitリポジトリを指定`
   - `// "Process Job DSLs" のビルドステップに相当`
   - `// SCMからチェックアウトしたファイルを実行する`

2. **インラインコメント**:
   - `targets('poc/jenkins/poc-job.groovy') // リポジトリ内のDSLファイルへのパス`

## 解決方法
すべての日本語コメントとインラインコメントを削除し、基本的なJob DSL構文のみを残しました。

### 修正前
```yaml
jobs:
  - script: >
      // この job {} ブロックはSeed Jobを定義します
      job('my-project-seed-job') {
        description('This seed job is created by JCasC. It reads DSL scripts from Git.')
        // ジョブ定義(DSL)を管理するGitリポジトリを指定
        scm {
          git {
            remote {
              url('https://github.com/kizuna-org/asr.git')
              // credentials('your-git-credentials-id')
            }
            branch('feat/poc')
          }
        }
        steps {
          // "Process Job DSLs" のビルドステップに相当
          jobDsl {
            // SCMからチェックアウトしたファイルを実行する
            targets('poc/jenkins/poc-job.groovy') // リポジトリ内のDSLファイルへのパス
            removedJobAction('DELETE')
            removedViewAction('DELETE')
          }
        }
      }
```

### 修正後
```yaml
jobs:
  - script: >
      job('my-project-seed-job') {
        description('This seed job is created by JCasC. It reads DSL scripts from Git.')
        scm {
          git {
            remote {
              url('https://github.com/kizuna-org/asr.git')
            }
            branch('feat/poc')
          }
        }
        steps {
          jobDsl {
            targets('poc/jenkins/poc_job.groovy')
            removedJobAction('DELETE')
            removedViewAction('DELETE')
          }
        }
      }
```

## 追加の問題と修正

### Job DSLファイル名エラー
Seed Jobが実行されたが、以下のエラーが発生：
```
ERROR: invalid script name 'poc-job.groovy; script names may only contain letters, digits and underscores, but may not start with a digit
```

**原因**: Job DSLファイル名にハイフン（`-`）が含まれていたため  
**解決方法**: ファイル名を`poc-job.groovy`から`poc_job.groovy`に変更

### 修正内容
- `poc/jenkins/poc-job.groovy` → `poc/jenkins/poc_job.groovy`
- `casc.yaml`内の参照も`targets('poc/jenkins/poc_job.groovy')`に更新

### Script Securityエラー
ファイル名修正後、新たなエラーが発生：
```
ERROR: script not yet approved for use
```

**原因**: JenkinsのScript Security機能により、Job DSLスクリプトが自動承認されない  
**解決方法**: JCasCでScript Securityを無効化

### Script Security設定追加
```yaml
# Script Security設定
security:
  globalJobDslSecurityConfiguration:
    useScriptSecurity: false
```

## Seed Job自動実行設定

### 設定追加の背景
初期設定では、Seed Jobは手動実行のみでした。実用的な運用のため、自動実行トリガーを追加しました。

### 追加したトリガー設定
```yaml
triggers {
  scm('H/5 * * * *')
  cron('H 2 * * *')
}
```

### トリガーの説明
1. **SCMポーリング** (`H/5 * * * *`)
   - 5分ごと（ランダム化された時間）にGitリポジトリをチェック
   - 変更があった場合に自動実行
   - Git commitやpushによって自動的にJob DSLが更新される

2. **定期実行** (`H 2 * * *`)
   - 毎日午前2時頃（ランダム化された時間）に実行
   - 変更がなくても定期的に実行してシステムの整合性を保つ
   - Job DSLファイルの外部削除などに対する復旧機能

## 結果
- Jenkins・Giteaコンテナが正常起動
- JCasCによるJob DSL設定が正常に読み込まれることを確認
- Seed Jobが正常実行され、Job DSLスクリプトが処理されることを確認
- Script Securityによる承認プロセスをスキップし、Job DSLが自動実行される
- **Gitリポジトリへの変更が自動的にJenkinsジョブに反映される**
- **手動実行不要でJob DSLスクリプトが定期的に処理される**

## 学習事項
- YAMLの`script: >`ブロック内はGroovyコードとして厳密に解釈される
- 日本語文字やUnicodeコメントはGroovyパーサーでエラーを引き起こす可能性
- JCascのデバッグは段階的なコメントアウトが効果的
- jenkins_homeディレクトリは手動作成せず、Dockerの自動作成に任せるべき
- **Job DSLファイル名は文字、数字、アンダースコアのみ使用可能（ハイフン不可）**
- **Job DSLファイル名は数字で始まってはいけない**
- **開発環境ではScript Securityを無効化することで自動化を促進できる**
- **本番環境では適切なScript Securityポリシーを検討する必要がある**
- **SCMポーリングによりGit-Jenkins間の自動同期が実現できる**
- **cronトリガーにより定期的なシステム整合性チェックが可能**

## 関連ファイル
- `poc/jenkins/casc.yaml`
- `poc/jenkins/poc_job.groovy` (旧: `poc-job.groovy`)
- `poc/compose.yaml`
