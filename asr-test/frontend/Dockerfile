# frontend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 依存関係をインストール
# run.shでrsyncしてからbuildするため、コンテキストはプロジェクトルートになる
COPY ./frontend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY ./frontend/app.py /app/

# ポートを開放 (ドキュメント用、実際にはcomposeでマッピング)
EXPOSE 8501
