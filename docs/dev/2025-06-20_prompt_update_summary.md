# 非推奨スクリプトの削除

**日付**: 2025-06-20  
**作業者**: Rovo Dev Assistant  
**概要**: 非推奨となっていたPub/Sub設定スクリプトの削除

## 背景

2025-06-18に行われたTerraformへの移行作業により、`infrastructure/setup-pubsub.sh`スクリプトは非推奨となっていました。移行期間が経過し、すべてのユーザーがTerraformを使用するようになったため、このスクリプトを削除することになりました。

## 実施内容

1. **非推奨スクリプトの削除**:
   - `infrastructure/setup-pubsub.sh`を削除

2. **ドキュメントの更新**:
   - `README.md`から非推奨スクリプトに関する記述を削除
   - ディレクトリ構造の説明から`setup-pubsub.sh`の記述を削除

## 影響

- 新規ユーザーが古い方法でセットアップしようとする可能性が排除されました
- コードベースがよりクリーンになりました
- 混乱の原因となる古いドキュメントが削除されました

## 関連ファイル

- 削除: `infrastructure/setup-pubsub.sh`
- 更新: `README.md`
- 追加: `docs/dev/2025-06-20_prompt_update_summary.md`（本ドキュメント）

## 関連する過去の変更

- [GCP設定のTerraform移行](2025-06-18_terraform_migration.md) - 2025-06-18に実施されたTerraformへの移行作業