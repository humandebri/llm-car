# llm-car

車の型式（型式コード／型式記号）からメーカー名・車名・排気量などを取得するための Gemini Grounding API 連携プロジェクトです。Google 検索によるグラウンディングを有効化した Gemini 2.0 Flash (experimental) モデルを用いて、回答に出典 URL を添えた構造化データを返します。

## 構成

- `gemini_car_lookup.py` : Gemini API を呼び出して車両情報を取得するモジュール
  - `GeminiCarLookupService` : 型式コードを渡すと `VehicleMatch` のリストを返すメインサービス
  - `lookup_car` : 単発の呼び出し向けヘルパー
- `lookup_example.py` : 型式コードを指定して動作確認し、トークン使用量から概算コストを表示するサンプルスクリプト
- `docs/gemini_car_lookup_design.md` : 設計メモ

## 前提条件

- Python 3.10 以上（開発環境は 3.13 を想定）
- Google AI Studio / Gemini API キー
- （任意）`uv` などでの仮想環境管理

## セットアップ

1. 依存関係（`requests` など）をインストールします。
   ```bash
   pip install -e .
   ```
   `uv` を利用する場合は `uv sync` で同様に依存関係を解決できます。

2. ルートディレクトリにある `.env` に Gemini API キーを設定します。
   ```env
   GEMINI_API_KEY=sk-xxxxxxxxxxxxxxxx
   ```
   `.env` はモジュール読み込み時に自動でロードされ、Git 管理対象外になっています。

## 使い方

### Python から呼び出す

```python
from gemini_car_lookup import lookup_car

result = lookup_car("DBA-ZRR70W")

for match in result.matches:
    print(match.to_dict())

print(result.usage_metadata)

# 英語で結果を受け取りたい場合
result_en = lookup_car("DBA-ZRR70W", response_language="English")
```

`usage_metadata` にはトークン消費量などが含まれるため、課金額の目安計算に利用できます。

### サンプルスクリプト

`.env` を設定済みなら、以下で CLI から確認できます。
```bash
python lookup_example.py
```

出力例（概略）：
- 型式に紐づく候補の辞書データ（メーカー・車名・排気量・根拠 URL など）
- Gemini の `usageMetadata` 情報
- 料金表に基づく概算コスト

### Streamlit デモ

ブラウザ上で試す場合は Streamlit アプリを起動します。

```bash
streamlit run streamlit_app.py
```

サイドバーで API キーを入力（または `.env` の値を利用）し、型式コードを指定して Lookup を実行できます。レスポンスの Raw JSON を表示するチェックボックスも用意しているので、解析に失敗した際のデバッグにも活用できます。

## トラブルシューティング

- **JSONDecodeError が出る**: Gemini が JSON の後ろに説明文を付ける場合があります。ライブラリ側である程度吸収しますが、それでも失敗する場合は `lookup_car(..., include_raw_response=True)` を指定して生のレスポンスを確認し、プロンプト調整を検討してください。
- **情報が取得できない**: 型式コードが正式なものであるか確認し、類似の型式を試すか、検索結果が少ない場合は `GeminiCarLookupService` の `system_instruction` や `temperature` を調整してください。

## コストの目安

`lookup_example.py` で使用量を `promptTokenCount` / `candidatesTokenCount` として取得し、Gemini 2.0 Flash (experimental) の暫定単価
(現時点の参考値: 入力 1,000 トークンあたり $0.00035、出力 1,000 トークンあたり $0.0007) を用いて概算を計算しています。実際の料金は必ず Google Cloud の最新料金ページで確認し、定数を更新してください。

## ライセンス

このリポジトリに特別なライセンスが設定されていない場合は、私用目的で利用してください。必要であればプロジェクトに適切なライセンスファイルを追加してください。
