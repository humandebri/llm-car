"""Streamlit demo for Gemini-based vehicle lookup."""

from __future__ import annotations

import os
from typing import Dict, Optional

import streamlit as st

from gemini_car_lookup import (
    GeminiCarLookupError,
    GeminiCarLookupService,
)

PROMPT_RATE_PER_1K_TOKENS_USD = 0.00035
OUTPUT_RATE_PER_1K_TOKENS_USD = 0.0007


def estimate_cost(usage: Optional[Dict[str, int]]) -> Optional[Dict[str, float]]:
    if not usage:
        return None

    prompt_tokens = usage.get("promptTokenCount", 0) or 0
    output_tokens = usage.get("candidatesTokenCount", 0) or 0

    prompt_cost = (prompt_tokens / 1000) * PROMPT_RATE_PER_1K_TOKENS_USD
    output_cost = (output_tokens / 1000) * OUTPUT_RATE_PER_1K_TOKENS_USD

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_cost": prompt_cost,
        "output_cost": output_cost,
        "total_cost": prompt_cost + output_cost,
    }


st.set_page_config(page_title="型式 Lookup", page_icon="🚗", layout="wide")

st.title("Gemini による型式 Lookup デモ")
st.caption("Gemini 2.0 Flash (experimental) + Google 検索グラウンディングで型式から車両情報を引き当てます")

with st.sidebar:
    st.header("設定")
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=default_api_key,
        placeholder="sk-...",
        type="password",
        help="環境変数 GEMINI_API_KEY を利用する場合は空欄でも構いません",
    )
    include_raw = st.checkbox(
        "レスポンスの raw JSON も表示",
        value=False,
        help="Gemini のレスポンス解析に失敗した際のデバッグ用",
    )
    response_language = st.selectbox(
        "レスポンス言語",
        options=["日本語", "English", "한국어", "繁體中文"],
        index=0,
        help="自由入力に切り替えたい場合は後でコードを編集してください",
    )
    st.markdown("---")
    st.caption(
        "料金はサンプル値で概算します。最新の料金表を必ず確認してください。"
    )

model_code = st.text_input("型式コード", placeholder="例: DBA-ZRR70W")
lookup_button = st.button("Lookup", type="primary")

if lookup_button:
    if not model_code.strip():
        st.warning("型式コードを入力してください。")
    else:
        with st.spinner("Gemini へ問い合わせ中..."):
            try:
                service = GeminiCarLookupService(
                    api_key=api_key_input or None,
                    include_raw_response=include_raw,
                    response_language=response_language,
                )
                result = service.lookup(model_code.strip())
            except ValueError as exc:
                st.error(f"初期化エラー: {exc}")
            except GeminiCarLookupError as exc:
                st.error(f"Lookup に失敗しました: {exc}")
                st.info("`レスポンスの raw JSON も表示` を有効化すると詳細を確認できます。")
            else:
                matches = [match.to_dict() for match in result.matches]
                if matches:
                    st.subheader("検索結果")
                    st.write(matches)
                else:
                    st.info("該当する車両情報が見つかりませんでした。")

                usage = result.usage_metadata
                estimated = estimate_cost(usage)
                st.subheader("Usage / Cost")
                if estimated:
                    st.metric(
                        label="推定コスト (USD)",
                        value=f"${estimated['total_cost']:.6f}",
                        delta=None,
                    )
                    st.write(
                        {
                            "prompt_tokens": estimated["prompt_tokens"],
                            "output_tokens": estimated["output_tokens"],
                            "prompt_cost_usd": round(estimated["prompt_cost"], 6),
                            "output_cost_usd": round(estimated["output_cost"], 6),
                        }
                    )
                else:
                    st.info("Usage metadata が返らなかったため、コストを算出できませんでした。")

                if include_raw and result.raw_response is not None:
                    with st.expander("Raw response"):
                        st.json(result.raw_response)

                st.success("Lookup 完了")
