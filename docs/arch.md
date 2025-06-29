# MLOps学習基盤 設計書

## 1\. 概要

本ドキュメントは、AIモデルの学習パイプラインを自動化・効率化するためのMLOps基盤の設計について定義する。

本基盤は、GitHub上のソースコード管理から、JenkinsによるCI/CD、Dockerによる環境分離、学習実行までの一連のフローを自動化することを目的とする。また、インフラ構成をコード (IaC/CaaC) としてリポジトリ内で一元管理することで、高い再現性とメンテナンス性を実現する。

## 2\. システム要件

本基盤は、以下の技術的要件および制約条件を満たすように設計する。

  * **実行環境:** 学習サーバーはDockerコンテナの実行のみを許可する。
  * **ソース管理:** GitHubを使用し、単一のモノレポでMLモデルコードとインフラ定義を管理する。
  * **CI/CD:** Jenkinsを使用し、Docker-out-of-Docker (DooD) 構成でコンテナのビルド・実行を行う。
  * **トリガー:** GitHubでリリース用Pull Requestがマージされたことを起点に、学習パイプラインを自動実行する。
  * **インフラ構成管理:**
      * サーバーのプロビジョニングはAnsibleによって行う。
      * Jenkins自体の設定はConfiguration as Code (CaaC) によってYAMLで管理する。
  * **外部アクセス:** サーバーは直接公開せず、リバースプロキシ(`frp`)を介してGitHubからのWebhookを受信する。
  * **運用・監視:** Docker環境の可視化と手動管理のため、Portainerを導入する。

## 3\. 全体アーキテクチャ

### 3.1. システム構成図

```mermaid
graph TD
    subgraph GitHub
        A[開発者がPRを作成・マージ] --> B(my-mlops-monorepo);
        B -- 1. Webhook Trigger --> C[frp Server (Public Cloud)];
    end

    subgraph "学習実行サーバー (On-Premise / Local)"
        D[Jenkins] -- CaaCで設定管理 --> E(infra/jenkins/casc.yaml);
        D -- 3. Run Pipeline --> F[Docker Daemon on Host];
        F -- DooD --> G[学習用コンテナを起動];
        C -- 2. Reverse Proxy --> D;
        H[Portainer] -- 管理/監視 --> F;
        I[frp Client] -- Proxy --> D;
    end

    subgraph "外部サービス"
        J[Hugging Face Hub];
    end

    G -- 4. データ・モデルのDL/UL --> J;
```

### 3.2. 処理フロー

1.  **トリガー**: 開発者が`my-mlops-monorepo`リポジトリでリリースPRをマージする。これをトリガーにGitHub Actionsが起動し、公開`frp`サーバーに対してWebhookを送信する。
2.  **リクエスト転送**: `frp`サーバーはリクエストを受け取り、ローカルで稼働する`frp`クライアントを経由して、学習サーバー上のJenkinsへ転送する。
3.  **パイプライン実行**: Jenkinsはリクエストを受け取り、リポジトリ内の`Jenkinsfile`に従ってパイプラインを開始する。Docker-out-of-Docker (DooD) 機構を利用して、ホストのDockerデーモンを操作し、対象プロジェクトの学習用コンテナをビルド・実行する。
4.  **学習処理**: 起動したコンテナ内のスクリプトが、Hugging Faceなどの外部サービスからデータセットやベースモデルをダウンロードし、学習を実行する。学習後の成果物（モデル、ログ等）は再び外部サービスへアップロードする。
5.  **監視**: 管理者および開発者は、PortainerのGUIを通じて、実行中のコンテナの状態、ログ、リソース使用状況をリアルタイムで監視する。

## 4\. リポジトリ設計

### 4.1. 基本方針

MLモデルのアプリケーションコードと、それを支えるインフラ構成定義を単一のモノレポで管理する。これにより、両者の整合性を保ち、システム全体の変更履歴を単一のソースで追跡可能にする。

### 4.2. ディレクトリ構造

```plaintext
my-mlops-monorepo/
├── .github/
│   └── workflows/
│       ├── release-trigger.yml  # Jenkinsへのトリガーに特化したAction
│       └── ci.yml               # (推奨) PR時のLinter/Test用Action
│
├── .gitignore
├── Jenkinsfile                  # 全プロジェクト共通のパイプライン定義
├── README.md                    # リポジトリ全体の概要と利用方法
│
├── projects/                    # 各AI学習プロジェクトのルートディレクトリ
│   ├── project-a/
│   │   ├── Dockerfile           # project-a専用の学習環境定義
│   │   ├── src/                 # 学習コード
│   │   ├── requirements.txt     # Python依存ライブラリ
│   │   ├── config.yaml          # ハイパーパラメータ等の設定ファイル
│   │   └── run.sh               # Jenkinsから呼び出す実行スクリプト
│   └── project-b/
│       └── ... (project-aと同様の構成)
│
└── infra/                       # インフラ設定を集約するディレクトリ
    ├── ansible/
    │   ├── playbook.yml         # ホストサーバーの初期設定
    │   └── ...
    │
    ├── docker-compose.yml       # Jenkins, Portainer, frp-client を起動するComposeファイル
    │
    ├── jenkins/
    │   └── casc.yaml            # Jenkins Configuration as Code 設定ファイル
    │
    ├── portainer/
    │   └── data/                # Portainerの永続化データ (.gitignore対象)
    │
    └── frp/
        └── frpc.ini             # frpクライアントの設定ファイル
```

### 4.3. 主要なファイルとディレクトリの役割

| パス | 役割 |
| --- | --- |
| `projects/` | 個別のAIモデル学習プロジェクトを格納する。プロジェクトごとにサブディレクトリを作成し、自己完結した構成とする。 |
| `projects/<name>/Dockerfile` | プロジェクト固有のライブラリや環境を定義する。これにより、プロジェクト間の依存関係の衝突を完全に防ぐ。 |
| `projects/<name>/run.sh` | Jenkinsから呼び出されるエントリーポイント。Pythonスクリプトの実行や前処理などをカプセル化し、`Jenkinsfile`をシンプルに保つ。 |
| `infra/` | 本基盤を支えるインフラサービスの定義を集約する。 |
| `infra/docker-compose.yml` | Jenkins, Portainer, frp-clientの3つのコアサービスを宣言的に定義し、`docker-compose up`コマンドで一括起動・管理する。 |
| `infra/jenkins/casc.yaml` | Jenkinsのジョブ定義、プラグイン設定、認証情報などをYAML形式で記述する。Jenkins本体の設定をコードとして管理する。 |
| `infra/ansible/` | 学習サーバーの初期構築（Docker, Docker Composeのインストール等）を行うためのAnsible Playbookを格納する。 |
| `.github/workflows/` | GitHub Actionsのワークフロー定義。本設計では、Jenkinsへのトリガー通知に特化した軽量な役割を担う。 |
| `Jenkinsfile` | リポジトリのルートに配置されるパイプライン定義。パラメータ（プロジェクト名など）を受け取り、動的に対象プロジェクトのビルドと実行を行う。 |

## 5\. ワークフロー

### 5.1. インフラ初期構築フロー (管理者向け)

1.  学習サーバーにログインし、本リポジトリをクローンする。
2.  `infra/ansible/` ディレクトリに移動し、`ansible-playbook playbook.yml` を実行して、サーバーの基本設定（Docker, Docker Composeのインストール等）を完了させる。
3.  `infra/` ディレクトリに移動し、`docker-compose up -d` を実行する。これにより、Jenkins, Portainer, frp-clientコンテナが起動する。
4.  Webブラウザで `http://<サーバーIP>:9000` にアクセスし、Portainerの管理者アカウントを初期設定する。

### 5.2. AIモデル学習実行フロー (開発者向け)

1.  `projects/<name>/` ディレクトリ内で、学習コードやDockerfileを開発・修正する。
2.  変更をコミットし、GitHubにPushしてPull Requestを作成する。
3.  （オプション）`ci.yml` が自動実行され、Linterや単体テストによるコード品質チェックが行われる。
4.  コードレビュー後、リリース担当者がリリース用Pull Requestをマージする。
5.  `release-trigger.yml` が起動し、Jenkinsのパイプラインをキックする。
6.  Jenkinsは`Jenkinsfile`に従い、対象プロジェクトのDockerイメージをビルドし、コンテナを起動して`run.sh`を実行する。
7.  学習が実行され、成果物はHugging Face等の外部サービスに保存される。

### 5.3. 運用・監視フロー

  * **Jenkins UI (`http://<サーバーIP>:8080`)**: パイプラインの実行履歴、ビルドログ、成功/失敗のステータスを確認する。
  * **Portainer UI (`http://<サーバーIP>:9000`)**:
      * 現在実行中の全コンテナ（Jenkins, 学習用コンテナ等）の一覧とリソース使用状況を監視する。
      * 各コンテナのリアルタイムログを閲覧し、問題のトラブルシューティングを行う。
      * 不要になったDockerイメージやボリュームを手動で削除する。

## 6\. 主要コンポーネント設定例

### 6.1. `infra/docker-compose.yml`

```yaml
version: '3.8'

services:
  jenkins:
    image: jenkins/jenkins:lts-jdk17
    container_name: jenkins
    ports:
      - "8080:8080"
    volumes:
      - jenkins_data:/var/jenkins_home
      - ./jenkins/casc.yaml:/var/jenkins_home/casc_configs/casc.yaml
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - CASC_JENKINS_CONFIG=/var/jenkins_home/casc_configs/casc.yaml
    restart: unless-stopped

  portainer:
    image: portainer/portainer-ce:latest
    container_name: portainer
    ports:
      - "9000:9000"
    volumes:
      - ./portainer/data:/data
      - /var/run/docker.sock:/var/run/docker.sock
    restart: unless-stopped

  frp_client:
    image: snowdreamtech/frpc:0.51.3
    container_name: frp-client
    volumes:
      - ./frp/frpc.ini:/etc/frp/frpc.ini
    network_mode: "host"
    restart: unless-stopped

volumes:
  jenkins_data:
```

### 6.2. `infra/jenkins/casc.yaml` (抜粋例)

```yaml
jobs:
  - script: >
      multibranchPipelineJob('my-mlops-monorepo') {
        branchSources {
          git {
            id = 'github-ml-monorepo'
            remote('https://github.com/<your-org>/my-mlops-monorepo.git')
            credentialsId('github-credentials-id')
          }
        }
      }
```
