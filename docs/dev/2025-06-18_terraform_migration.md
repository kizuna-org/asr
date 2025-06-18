# GCP設定のTerraform移行

**日付**: 2025-06-18  
**作業者**: Rovo Dev Assistant  
**概要**: GCP Pub/Sub設定をbashスクリプトからTerraformに移行

## 背景

従来、GCP Pub/Subの設定は `infrastructure/setup-pubsub.sh` というbashスクリプトで行っていましたが、Infrastructure as Code (IaC) のベストプラクティスに従い、Terraformに移行することになりました。

## 実装内容

### 1. Terraform設定ファイルの作成

以下のファイルを `infrastructure/terraform/` ディレクトリに作成しました：

- **main.tf**: メインのTerraform設定
  - Pub/Subトピック（build-triggers, app-triggers）
  - Pub/Subサブスクリプション
  - GitHub Actions用サービスアカウント
  - IAMポリシーバインディング
  - サービスアカウントキー

- **variables.tf**: 変数定義
  - project_id, region, environment
  - トピック名、サブスクリプション名
  - サービスアカウント名

- **outputs.tf**: 出力定義
  - 作成されたリソースの情報
  - GitHub Secrets設定手順

- **terraform.tfvars.example**: 設定例ファイル

- **README.md**: Terraform使用方法とマイグレーション手順

- **.gitignore**: Terraformファイル用の除外設定

### 2. 既存bashスクリプトの非推奨化

`infrastructure/setup-pubsub.sh` に以下の変更を加えました：

- 非推奨警告メッセージの追加
- Terraformの使用を促すメッセージ
- 実行前の確認プロンプト

### 3. ドキュメントの更新

`README.md` を更新して：

- Terraformの使用を推奨する記述に変更
- bashスクリプトを非推奨として明記
- ディレクトリ構造にTerraform設定を追加

## 利点

### Infrastructure as Code (IaC)
- 設定がコードとして管理される
- バージョン管理が可能
- 変更履歴の追跡が容易

### 宣言的設定
- 現在の状態と期待する状態の差分を自動で検出
- 冪等性が保証される
- 設定ドリフトの検出が可能

### 再現性
- 同じ設定を複数の環境で再現可能
- 災害復旧時の迅速な復元
- 開発・ステージング・本番環境の一貫性

### 保守性
- 設定の可読性向上
- モジュール化による再利用性
- 依存関係の明確化

## マイグレーション手順

### 新規セットアップの場合

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
# terraform.tfvarsを編集
terraform init
terraform plan
terraform apply
```

### 既存環境からの移行

1. **既存リソースのインポート**:
   ```bash
   terraform import google_pubsub_topic.build_triggers projects/PROJECT_ID/topics/build-triggers
   terraform import google_pubsub_topic.app_triggers projects/PROJECT_ID/topics/app-triggers
   # 他のリソースも同様にインポート
   ```

2. **状態の確認**:
   ```bash
   terraform plan
   ```

3. **必要に応じて適用**:
   ```bash
   terraform apply
   ```

## 今後の予定

1. **既存bashスクリプトの完全削除**: 十分な移行期間後に削除予定
2. **Terraform Cloudの検討**: 状態管理の改善
3. **モジュール化**: 他のGCPリソースもTerraform化

## 関連ファイル

- `infrastructure/terraform/` - 新しいTerraform設定
- `infrastructure/setup-pubsub.sh` - 非推奨化されたbashスクリプト
- `README.md` - 更新されたセットアップ手順
- `docs/architecture.md` - システム全体のアーキテクチャ

## 注意事項

- 既存環境を移行する際は、必ず事前にバックアップを取得してください
- Terraformの状態ファイル（.tfstate）は機密情報を含むため、適切に管理してください
- サービスアカウントキーは引き続き機密情報として扱い、GitHub Secretsに安全に保存してください