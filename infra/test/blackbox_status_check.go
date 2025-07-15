package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
)

// Blackbox exporterのレスポンスの一部
// { "status": "success", "data": { "result": [ { "metric": { ... }, "value": [ <timestamp>, "1" ] } ] } }
type BlackboxResponse struct {
	Status string `json:"status"`
	Data   struct {
		Result []struct {
			Metric map[string]string `json:"metric"`
			Value  []interface{}     `json:"value"`
		} `json:"result"`
	} `json:"data"`
}

func main() {
	url := flag.String("url", "", "Blackbox exporterのPrometheus APIエンドポイントURL (例: http://localhost:9115/probe?target=example.com&module=http_2xx)")
	flag.Parse()

	if *url == "" {
		fmt.Fprintln(os.Stderr, "URLを指定してください (--url)")
		os.Exit(1)
	}

	resp, err := http.Get(*url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "HTTPリクエスト失敗: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "レスポンス読み込み失敗: %v\n", err)
		os.Exit(1)
	}

	var result BlackboxResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Fprintf(os.Stderr, "JSONパース失敗: %v\n", err)
		os.Exit(1)
	}

	if result.Status != "success" || len(result.Data.Result) == 0 {
		fmt.Fprintln(os.Stderr, "Prometheus APIのステータスがsuccessでない、または結果が空です")
		os.Exit(1)
	}

	// 通常は value[1] が "1" ならOK (up=1)
	valueStr, ok := result.Data.Result[0].Value[1].(string)
	if !ok {
		fmt.Fprintln(os.Stderr, "value[1]が文字列でありません")
		os.Exit(1)
	}

	if valueStr == "1" {
		fmt.Println("UP (1)")
		os.Exit(0)
	} else {
		fmt.Printf("DOWN (%s)\n", valueStr)
		os.Exit(1)
	}
}
