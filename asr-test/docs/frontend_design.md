# フロントエンド設計書

このドキュメントは、Streamlitを使用して構築されるWeb GUIのコンポーネント、状態管理、およびバックエンドとの通信について詳細に設計します。

## 1. 画面構成

画面は主に以下のセクションで構成されます。

1.  **サイドバー (Control Panel)**
    -   モデル選択 (`st.selectbox`)
    -   データセット選択 (`st.selectbox`)
    -   学習開始ボタン (`st.button`)
    -   学習停止ボタン (`st.button`)
    -   現在のステータス表示 (`st.info` or `st.success`)
2.  **メインエリア (Dashboard)**
    -   **学習グラフエリア**:
        -   学習/検証ロスをプロットするグラフ (`st.line_chart`)
        -   学習率の推移をプロットするグラフ (`st.line_chart`)
    -   **進捗表示エリア**:
        -   現在のエポック/ステップ (`st.progress`)
        -   最新のロス値 (`st.metric`)
    -   **ログ出力エリア**:
        -   バックエンドからのログメッセージを表示するテキストエリア (`st.text_area`)
    -   **推論テストエリア**:
        -   音声ファイルアップローダー (`st.file_uploader`)
        -   推論実行ボタン (`st.button`)
        -   推論結果表示 (`st.text_area`)

## 2. 状態管理 (st.session_state)

UIの状態とバックエンドから受信したデータは `st.session_state` を使って管理します。

```python
# app.py (擬似コード)

# 初期化
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'progress_df' not in st.session_state:
    st.session_state.progress_df = pd.DataFrame(columns=['step', 'loss', 'lr'])
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
```

-   `is_training` (bool): 学習が実行中かどうか。UIの有効/無効を切り替えるために使用。
-   `logs` (list[str]): バックエンドから受信したログメッセージのリスト。
-   `progress_df` (pd.DataFrame): グラフ描画用のデータ。`step`, `loss`, `lr` などのカラムを持つ。
-   `ws_client` (WebSocketClient): WebSocketクライアントのインスタンス。接続管理に使用。

## 3. バックエンドとの通信フロー

### 3.1. 学習開始フロー

1.  **[UI]** ユーザーがサイドバーでモデルとデータセットを選択し、「学習開始」ボタンをクリックします。
2.  **[app.py]** `on_click` コールバックがトリガーされます。
3.  **[app.py]** `requests.post` を使用して、バックエンドの `/train/start` エンドポイントに `{ "model_name": ..., "dataset_name": ... }` を送信します。
4.  **[app.py]** レスポンスが `202 Accepted` であれば、`st.session_state.is_training = True` に設定します。
5.  **[app.py]** WebSocket接続処理を開始します (`connect_to_websocket` 関数を呼び出す)。
6.  **[app.py]** `st.experimental_rerun()` を呼び出してUIを更新し、ボタンを無効化したり、ステータス表示を変更したりします。

### 3.2. WebSocketによる進捗受信フロー

`connect_to_websocket` 関数は非同期で実行され、バックエンドからのメッセージを待ち受けます。接続が切断された場合は、自動的に再接続を試みます。

```python
# app.py (擬似コード)
async def websocket_listener():
    retry_count = 0
    max_retries = 5
    retry_delay = 3 # seconds

    while st.session_state.is_training and retry_count < max_retries:
        try:
            async with websockets.connect("ws://backend/ws") as ws:
                st.session_state.ws_client = ws
                retry_count = 0 # 接続成功でリトライカウントをリセット
                st.session_state.logs.append("INFO: WebSocketに接続しました。")
                st.experimental_rerun()

                while st.session_state.is_training:
                    message = await ws.recv()
                    data = json.loads(message)
                    handle_ws_message(data)
                    st.experimental_rerun()

        except (websockets.ConnectionClosed, OSError) as e:
            st.session_state.logs.append(f"WARNING: WebSocket接続が切れました: {e}")
            retry_count += 1
            if retry_count < max_retries:
                st.session_state.logs.append(f"INFO: {retry_delay}秒後に再接続します... ({retry_count}/{max_retries})")
                st.experimental_rerun()
                await asyncio.sleep(retry_delay)
            else:
                st.session_state.logs.append("ERROR: WebSocketの再接続に失敗しました。")
                st.session_state.is_training = False
                st.experimental_rerun()
                break

def handle_ws_message(data):
    if data['type'] == 'progress':
        # st.session_state.progress_df にデータを追加
    elif data['type'] == 'log':
        # st.session_state.logs にメッセージを追加
    elif data['type'] == 'status' and data['payload']['status'] in ['completed', 'stopped']:
        st.session_state.is_training = False
    elif data['type'] == 'error':
        # エラーメッセージをログに追加
        st.session_state.is_training = False
```

**フローの更新点:**

-   **再接続ループ:** `websocket_listener` 全体が `while` ループで囲われ、最大試行回数 (`max_retries`) に達するまで接続を試みます。
-   **例外処理:** `websockets.ConnectionClosed` や `OSError` (ネットワーク起因のエラー) を捕捉します。
-   **リトライ処理:** 接続が切れた場合、ログに警告を出し、一定時間 (`retry_delay`) 待ってから再接続を試みます。
-   **リトライ失敗:** 最大試行回数に達しても接続できない場合は、エラーを記録し、学習状態 (`is_training`) を `False` にしてプロセスを終了します。

### 3.3. 推論フロー

1.  **[UI]** ユーザーが `st.file_uploader` を使って音声ファイルをアップロードします。
2.  **[UI]** 「推論実行」ボタンをクリックします。
3.  **[app.py]** `requests.post` を使用して、バックエンドの `/inference` エンドポイントに `multipart/form-data` としてファイルを送信します。
4.  **[app.py]** レスポンス (JSON) を受け取り、`transcription` の値を抽出します。
5.  **[app.py]** 結果を `st.text_area` に表示します。
