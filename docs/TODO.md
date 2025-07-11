# TODO List

## CI/CD パイプライン改善

### 🔄 Git-Jenkins統合の最適化

#### 現状
- **SCMポーリング方式**: 5分ごとにGitリポジトリをチェック
- **問題点**:
  - レスポンス遅延（最大5分）
  - 不要なポーリングによるリソース消費
  - スケーラビリティの制限

#### 改善案：Push型通知の実装

##### 1. GitHub Webhooks
- [ ] GitHub Repository設定でWebhook追加
- [ ] Jenkins Webhook URLの設定
- [ ] GitHub Plugin設定とトリガー構成
- [ ] セキュリティ設定（Secret Token）

##### 2. Gitea Webhooks（ローカル開発用）
- [ ] Gitea WebhookのJenkins連携設定
- [ ] ローカル環境でのWebhook受信確認
- [ ] ngrokまたは類似ツールでの開発環境テスト

##### 3. Generic Webhook Plugin
- [ ] Generic Webhook Triggerプラグインの導入
- [ ] カスタムWebhookエンドポイントの設定
- [ ] 複数Gitプラットフォーム対応

### 🚀 Jenkins起動時実行の実装

#### 現状の課題
- [ ] JCaSC内でのスクリプト自動実行は構文エラーを引き起こす
- [ ] 複数jobスクリプトの実行順序制御が困難
- [ ] Jenkins初期化完了タイミングの検出が複雑

#### 実装候補

##### 1. Docker Compose ヘルスチェック連携
- [ ] Jenkinsコンテナのヘルスチェック実装
- [ ] 起動完了後のcurlによるSeed Job実行
- [ ] docker-compose.ymlの設定追加

```yaml
# 実装例
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/login"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 90s
```

##### 2. Init Script（推奨）
- [ ] `/usr/share/jenkins/ref/init.groovy.d/`に起動スクリプト配置
- [ ] Dockerfileでの自動配置設定
- [ ] Jenkins初期化完了後の自動実行

```groovy
// 実装例
import jenkins.model.Jenkins

def jenkins = Jenkins.getInstance()
Thread.start {
  Thread.sleep(10000) // 初期化完了待機
  def job = jenkins.getItem('my-project-seed-job')
  if (job) {
    job.scheduleBuild2(0)
    println "Seed job triggered automatically"
  }
}
```

##### 3. External Trigger Script
- [ ] Bash/Python外部スクリプトの作成
- [ ] Jenkins REST APIによる実行
- [ ] Crontabまたはsystemdタイマーでの定期チェック

##### 4. Jenkins Plugin開発
- [ ] カスタムプラグインでの起動時実行
- [ ] JCasCとの完全統合
- [ ] より高度な制御とログ機能

#### 実装優先度
1. **Init Script** - 最も確実で標準的
2. **Docker Healthcheck** - インフラレベルで制御
3. **External Script** - シンプルで柔軟
4. **Custom Plugin** - 最も高機能だが開発コスト大

### 📈 パフォーマンス向上

#### 現状の課題
- [ ] ポーリング間隔の最適化検討
- [ ] 大規模リポジトリでの効率化
- [ ] 複数ブランチ対応の改善

#### 実装予定
- [ ] 即座のトリガー（Push → 即時実行）
- [ ] 差分チェックによる不要実行の削減
- [ ] 並列実行による処理時間短縮

### 🔒 セキュリティ強化

#### Webhook Security
- [ ] Secret Token認証の実装
- [ ] IP Allowlist設定
- [ ] HTTPS必須化

#### Jenkins Security
- [ ] Webhook受信の権限制御
- [ ] 不正実行の防止メカニズム
- [ ] ログ監視とアラート

### 🛠️ 運用改善

#### モニタリング
- [ ] Webhook受信状況の監視
- [ ] 実行失敗時のアラート
- [ ] パフォーマンスメトリクス取得

#### デバッグ・トラブルシューティング
- [ ] Webhook受信ログの詳細化
- [ ] 実行状況の可視化
- [ ] 障害時の自動復旧メカニズム

### 📋 実装優先度

#### High Priority
1. **Jenkins起動時実行（Init Script）** - 手動実行の自動化
2. **GitHub Webhooks** - 本番環境での即時反映
3. **Generic Webhook Plugin** - 汎用性確保
4. **セキュリティ設定** - 安全な運用

#### Medium Priority
1. **Docker Healthcheck連携** - インフラ統合
2. **Gitea Webhooks** - 開発環境の改善
3. **パフォーマンス最適化** - 大規模対応
4. **モニタリング強化** - 運用品質向上

#### Low Priority
1. **高度な並列化** - 将来的な拡張
2. **マルチクラウド対応** - スケールアウト
3. **AI連携** - 自動最適化

### 📝 関連ドキュメント

- [ ] Webhook設定手順書の作成
- [ ] 起動時実行実装ガイド
- [ ] トラブルシューティングガイド
- [ ] セキュリティベストプラクティス
- [ ] パフォーマンスチューニングガイド

### 🧪 テスト計画

#### 機能テスト
- [ ] Webhook受信テスト
- [ ] 起動時実行テスト
- [ ] フォールバック機能テスト
- [ ] セキュリティテスト

#### パフォーマンステスト
- [ ] 高負荷時のWebhook処理
- [ ] 大量コミット時の対応
- [ ] 同時実行制限テスト

#### 運用テスト
- [ ] 障害復旧テスト
- [ ] メンテナンス時の動作確認
- [ ] ログローテーション確認

---

## 備考

- 現在のSCMポーリング設定は後方互換性として保持
- Push型実装後も、フォールバック機能として機能
- 段階的移行により安全な実装を目指す
- Jenkins起動時実行は安定性確保後に追加実装 
