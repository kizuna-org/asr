// WebSocket接続テスト用のJavaScriptコード
console.log("WebSocket接続テストを開始します...");

const ws = new WebSocket('ws://localhost:58081/ws');

ws.onopen = function (event) {
  console.log("✅ WebSocket接続が成功しました！");

  // 開始メッセージを送信
  const startMessage = {
    type: "start",
    model_name: "conformer",
    sample_rate: 16000,
    format: "i16"
  };
  ws.send(JSON.stringify(startMessage));
  console.log("📤 開始メッセージを送信:", startMessage);

  // 5秒後に接続を閉じる
  setTimeout(() => {
    ws.close();
    console.log("🔌 接続を閉じました");
  }, 5000);
};

ws.onmessage = function (event) {
  console.log("📥 メッセージを受信:", event.data);
};

ws.onerror = function (error) {
  console.error("❌ WebSocketエラー:", error);
};

ws.onclose = function (event) {
  console.log("🔌 WebSocket接続が閉じられました - Code:", event.code, "Reason:", event.reason);
};

