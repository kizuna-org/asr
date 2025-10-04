# WebRTC リアルタイム音声認識 - 直接アクセスガイド

## 問題の概要

WebRTCは**ピアツーピア (P2P)** 接続技術であり、ブラウザとサーバー間で直接UDP通信を確立する必要があります。

SSH port forwarding (`localhost:58080`) 経由でアクセスすると、WebRTC の ICE (Interactive Connectivity Establishment) 接続が失敗します。

### なぜlocalhost経由では動作しないのか?

1. **ブラウザ**: `localhost:58080` でStreamlitにアクセス
2. **WebRTC**: ブラウザが実際のネットワーク候補 (192.168.x.x, 公開IP) を収集
3. **Streamlitコンテナ**: Docker内部ネットワーク (172.20.0.x) から候補に到達不可
4. **結果**: ICE接続がタイムアウト (60秒後にFAILED)

## 解決方法: サーバーIPに直接アクセス

### ステップ1: サーバーIPを確認

```bash
cd /Users/5ouma/ghq/github.com/kizuna-org/asr/asr-test
./get-direct-url.sh
```

出力例:
```
✅ サーバーIP: 172.16.98.181

📱 WebRTC機能を使用するには、以下のURLに**直接**アクセスしてください:

   🌐 フロントエンド (Streamlit): http://172.16.98.181:58080
   🔧 バックエンド API:           http://172.16.98.181:58081
```

### ステップ2: ブラウザで直接アクセス

**重要**: `localhost`ではなく、上記のIPアドレスを使用してください！

```
http://172.16.98.181:58080
```

### ステップ3: リアルタイム音声認識を実行

1. ブラウザでマイクアクセスを許可
2. **START** ボタンをクリック (マイクをON)
3. **リアルタイム開始** ボタンをクリック
4. 🎤 話してください！

## トラブルシューティング

### ICE接続が失敗する場合

#### 確認1: 正しいURLでアクセスしているか?
```bash
# ブラウザのアドレスバーを確認:
# ❌ http://localhost:58080     <- これはダメ
# ✅ http://172.16.98.181:58080  <- これが正しい
```

#### 確認2: ファイアウォール設定

サーバー側でポートが開放されているか確認:
```bash
ssh edu-gpu "sudo firewall-cmd --list-ports"
# または
ssh edu-gpu "sudo ufw status"
```

必要なポート:
- **TCP 58080**: Streamlit frontend
- **TCP 58081**: FastAPI backend  
- **UDP 範囲**: WebRTC メディアトラフィック (動的に割り当て)

ポートを開放:
```bash
# firewalld の場合
ssh edu-gpu "sudo firewall-cmd --permanent --add-port=58080/tcp"
ssh edu-gpu "sudo firewall-cmd --permanent --add-port=58081/tcp"
ssh edu-gpu "sudo firewall-cmd --reload"

# ufw の場合
ssh edu-gpu "sudo ufw allow 58080/tcp"
ssh edu-gpu "sudo ufw allow 58081/tcp"
```

#### 確認3: ネットワーク接続

ローカルマシンからサーバーIPに到達できるか:
```bash
ping 172.16.98.181
curl http://172.16.98.181:58080
```

### マイクが動作しない場合

1. ブラウザのマイク権限を確認
2. システムのマイク設定を確認
3. 別のブラウザで試す (Chrome推奨)

### 音声フレームが届かない場合

バックエンドログを確認:
```bash
ssh edu-gpu "sudo docker logs asr-test-asr-api-1" | grep -E '\[WS\]|audio frame'
```

正常な場合の出力:
```
[WS] ✅ Setup complete, ready to receive audio frames
[WS] 🎤 First audio frame received! size=8192 bytes
[WS] 📊 Processing audio chunk: 8192 bytes
```

## 技術的詳細

### WebRTC ICE 接続フロー

1. **Offer/Answer交換**: SDP (Session Description Protocol) をやり取り
2. **ICE候補収集**: 
   - Host candidates (ローカルIP)
   - Server reflexive candidates (STUN経由の公開IP)
   - Relay candidates (TURN経由 - 今回は未使用)
3. **接続性チェック**: 各候補ペアでSTUNバインディング
4. **接続確立**: 最適なペアで UDP/RTP 通信開始

### 修正履歴

#### Issue 1: `asyncio` イベントループ競合
- **原因**: `asyncio.new_event_loop()` + `set_event_loop()` が streamlit-webrtc と競合
- **修正**: `asyncio.run()` を使用して適切なライフサイクル管理

#### Issue 2: Streamlit rerun による WebRTC 再作成
- **原因**: 毎回の rerun で `webrtc_streamer()` が呼ばれ ICE がリセット
- **修正**: `st.session_state` に `rtc_ctx` をキャッシュして再利用

#### Issue 3: ICE 接続タイムアウト (本質的問題)
- **原因**: SSH tunnel 経由では P2P 接続が確立できない
- **修正**: サーバーIPに直接アクセス (本ガイド)

## 参考リソース

- [WebRTC ICE Candidate Types](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceCandidate)
- [streamlit-webrtc Documentation](https://github.com/whitphx/streamlit-webrtc)
- [Interactive Connectivity Establishment (ICE)](https://datatracker.ietf.org/doc/html/rfc8445)

---

最終更新: 2025-10-02
