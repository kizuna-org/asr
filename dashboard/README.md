# 🚀 Rovo Dev CI/CD Dashboard

React + Vite + Tailwind CSSで構築されたモダンなCI/CDパイプライン監視ダッシュボード

## 🌟 特徴

- **リアルタイム監視**: パイプラインの状況をリアルタイムで表示
- **モダンUI**: Tailwind CSSによるスタイリッシュなデザイン
- **レスポンシブ**: モバイル・タブレット・デスクトップ対応
- **高速**: Viteによる高速な開発・ビルド環境
- **型安全**: TypeScript対応（JSXで実装）

## 🚀 クイックスタート

### 開発環境

```bash
# 依存関係のインストール
npm install

# 開発サーバー起動
npm run dev
```

### 本番ビルド

```bash
# ビルド実行
npm run build

# ビルド結果のプレビュー
npm run preview
```

## 📁 プロジェクト構造

```
dashboard/
├── src/
│   ├── components/          # Reactコンポーネント
│   │   ├── Header.jsx      # ヘッダーコンポーネント
│   │   ├── JobGrid.jsx     # ジョブ一覧グリッド
│   │   ├── JobCard.jsx     # 個別ジョブカード
│   │   ├── LoadingSpinner.jsx
│   │   ├── ErrorMessage.jsx
│   │   └── RefreshButton.jsx
│   ├── services/
│   │   └── api.js          # API通信サービス
│   ├── App.jsx             # メインアプリケーション
│   ├── main.jsx           # エントリーポイント
│   └── index.css          # グローバルスタイル
├── public/                 # 静的ファイル
├── package.json
├── vite.config.js         # Vite設定
├── tailwind.config.js     # Tailwind CSS設定
└── postcss.config.js      # PostCSS設定
```

## 🎨 デザインシステム

### カラーパレット

- **Primary**: Indigo/Blue グラデーション
- **Success**: Green系（成功状態）
- **Warning**: Amber系（実行中状態）
- **Error**: Red系（エラー状態）
- **Neutral**: Slate系（テキスト・背景）

### コンポーネント

- **Glass Effect**: 半透明のガラス風エフェクト
- **Card Hover**: ホバー時のアニメーション
- **Status Badge**: ステータス表示バッジ

## 📊 ログスキーマ対応

`schemas/log_schema.json`で定義されたログフォーマットに対応：

- **timestamp**: ISO 8601形式のタイムスタンプ
- **level**: DEBUG, INFO, WARN, ERROR, FATAL
- **component**: app, build_subscriber, app_subscriber, github_actions
- **context**: 操作コンテキスト、進捗情報
- **error**: エラー詳細情報
- **performance**: パフォーマンスメトリクス

## 🌐 Cloudflare Pages デプロイ

### 自動デプロイ設定

1. Cloudflare Pagesでリポジトリを接続
2. ビルド設定:
   ```
   Build command: npm run build
   Build output directory: dist
   Root directory: dashboard
   ```

### 環境変数（必要に応じて）

```bash
# R2 API設定（将来の実装用）
VITE_R2_ENDPOINT=your-r2-endpoint
VITE_R2_BUCKET=your-bucket-name
```

## 🔧 カスタマイズ

### API統合

現在はモックデータを使用していますが、`src/services/api.js`を編集してCloudflare R2との実際の統合を実装できます。

### スタイリング

`tailwind.config.js`でカラーテーマやアニメーションをカスタマイズ可能です。

## 🤝 開発ガイドライン

- コンポーネントは機能ごとに分割
- Tailwind CSSのユーティリティクラスを活用
- アクセシビリティを考慮した実装
- レスポンシブデザインの維持

## 📝 TODO

- [ ] Cloudflare R2との実際の統合
- [ ] リアルタイム更新（WebSocket/SSE）
- [ ] ジョブ詳細モーダル
- [ ] フィルタリング・検索機能
- [ ] エクスポート機能
- [ ] 通知システム統合