#!/usr/bin/env python3
"""
学習再開機能のテストスクリプト
"""

import requests
import json
import time
import sys

# API ベースURL
BASE_URL = "http://localhost:8000/api"

def test_api_endpoint(endpoint, method="GET", data=None):
    """APIエンドポイントをテストする"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"{method} {endpoint}: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
        else:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    print("=== 学習再開機能テスト ===")
    
    # 1. 設定情報を取得
    print("\n1. 設定情報を取得")
    test_api_endpoint("/config")
    
    # 2. 現在のステータスを確認
    print("\n2. 現在のステータスを確認")
    test_api_endpoint("/status")
    
    # 3. 利用可能なチェックポイントを確認
    print("\n3. 利用可能なチェックポイントを確認")
    test_api_endpoint("/checkpoints")
    
    # 4. 特定のモデルのチェックポイントを確認
    print("\n4. conformerモデルのチェックポイントを確認")
    test_api_endpoint("/checkpoints?model_name=conformer&dataset_name=ljspeech")
    
    # 5. 学習を開始（チェックポイントから再開）
    print("\n5. 学習を開始（チェックポイントから再開）")
    training_params = {
        "model_name": "conformer",
        "dataset_name": "ljspeech",
        "epochs": 2,
        "batch_size": 8,
        "resume_from_checkpoint": True,
        "lightweight": True
    }
    response = test_api_endpoint("/train/start", "POST", training_params)
    
    if response and response.status_code == 200:
        print("学習が開始されました。進捗を監視します...")
        
        # 6. 進捗を監視
        for i in range(10):
            time.sleep(5)
            print(f"\n進捗確認 {i+1}/10:")
            test_api_endpoint("/progress")
            
            # 学習が完了したかチェック
            status_response = test_api_endpoint("/status")
            if status_response and status_response.json().get("is_training") == False:
                print("学習が完了しました。")
                break
    
    # 7. 学習再開APIをテスト
    print("\n7. 学習再開APIをテスト")
    resume_params = {
        "model_name": "conformer",
        "dataset_name": "ljspeech",
        "epochs": 1,
        "batch_size": 8,
        "lightweight": True
    }
    response = test_api_endpoint("/train/resume", "POST", resume_params)
    
    if response and response.status_code == 200:
        print("学習再開が開始されました。進捗を監視します...")
        
        # 8. 再開後の進捗を監視
        for i in range(5):
            time.sleep(5)
            print(f"\n再開後の進捗確認 {i+1}/5:")
            test_api_endpoint("/progress")
            
            # 学習が完了したかチェック
            status_response = test_api_endpoint("/status")
            if status_response and status_response.json().get("is_training") == False:
                print("学習再開が完了しました。")
                break
    
    # 9. 最終的なチェックポイント一覧を確認
    print("\n9. 最終的なチェックポイント一覧を確認")
    test_api_endpoint("/checkpoints")
    
    # 10. フロントエンド機能のテスト
    print("\n10. フロントエンド機能のテスト")
    print("フロントエンドの学習再開機能をテストするには、以下の手順を実行してください：")
    print("1. フロントエンドを起動: streamlit run frontend/app.py")
    print("2. ブラウザで http://localhost:8501 にアクセス")
    print("3. 以下の機能をテスト:")
    print("   - チェックポイント管理ページでチェックポイント一覧を確認")
    print("   - 特定のチェックポイントから学習を再開")
    print("   - 学習再開時のグラフ表示を確認")
    print("   - 学習再開情報の表示を確認")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()
