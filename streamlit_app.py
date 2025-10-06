"""Streamlit demo for Gemini-based vehicle lookup."""

from __future__ import annotations

import os
from typing import Dict, Optional
from time import perf_counter

import streamlit as st

from gemini_car_lookup import (
    GeminiCarLookupError,
    GeminiCarLookupService,
)

PROMPT_RATE_PER_1K_TOKENS_USD = 0.0001
OUTPUT_RATE_PER_1K_TOKENS_USD = 0.0001
USD_TO_JPY_RATE = 150.0


def resolve_default_api_key() -> str:
    """Prefer Streamlit secrets over environment variables for the API key."""
    if "GEMINI_API_KEY" in st.secrets:
        secret_value = st.secrets["GEMINI_API_KEY"]
        if isinstance(secret_value, str):
            stripped = secret_value.strip()
            if stripped:
                return stripped
    return os.getenv("GEMINI_API_KEY", "").strip()


def estimate_cost(usage: Optional[Dict[str, int]]) -> Optional[Dict[str, float]]:
    if not usage:
        return None

    prompt_tokens = usage.get("promptTokenCount", 0) or 0
    output_tokens = usage.get("candidatesTokenCount", 0) or 0

    prompt_cost = (prompt_tokens / 1000) * PROMPT_RATE_PER_1K_TOKENS_USD
    output_cost = (output_tokens / 1000) * OUTPUT_RATE_PER_1K_TOKENS_USD
    total_cost = prompt_cost + output_cost

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_cost": prompt_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "prompt_cost_jpy": prompt_cost * USD_TO_JPY_RATE,
        "output_cost_jpy": output_cost * USD_TO_JPY_RATE,
        "total_cost_jpy": total_cost * USD_TO_JPY_RATE,
    }


st.set_page_config(page_title="型式 Lookup", page_icon="🚗", layout="wide")

st.title("Gemini による型式 Lookup デモ")
st.caption("Gemini 2.5 Flash + Google 検索グラウンディングで型式から車両情報を引き当てます")

with st.sidebar:
    st.header("設定")
    default_api_key = resolve_default_api_key()
    api_key_input = st.text_input(
        "Gemini API Key",
        value="",
        placeholder="sk-...",
        type="password",
        help="Streamlit の secrets または環境変数 GEMINI_API_KEY が設定済みなら空欄で OK",
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
            elapsed_seconds: Optional[float] = None
            try:
                effective_api_key = (api_key_input or default_api_key).strip() or None
                service = GeminiCarLookupService(
                    api_key=effective_api_key,
                    include_raw_response=include_raw,
                    response_language=response_language,
                )
                start_time = perf_counter()
                result = service.lookup(model_code.strip())
                elapsed_seconds = perf_counter() - start_time
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
                if elapsed_seconds is not None:
                    st.metric(
                        label="API 応答時間",
                        value=f"{elapsed_seconds:.2f} 秒",
                    )
                if estimated:
                    st.metric(
                        label="推定コスト",
                        value=f"${estimated['total_cost']:.6f}",
                        delta=f"約 ¥{estimated['total_cost_jpy']:.2f}",
                    )
                    st.write(
                        {
                            "prompt_tokens": estimated["prompt_tokens"],
                            "output_tokens": estimated["output_tokens"],
                            "prompt_cost_usd": round(estimated["prompt_cost"], 6),
                            "output_cost_usd": round(estimated["output_cost"], 6),
                            "prompt_cost_jpy": round(estimated["prompt_cost_jpy"], 2),
                            "output_cost_jpy": round(estimated["output_cost_jpy"], 2),
                            "usd_to_jpy": USD_TO_JPY_RATE,
                            "total_cost_jpy": round(estimated["total_cost_jpy"], 2),
                        }
                    )
                else:
                    st.info("Usage metadata が返らなかったため、コストを算出できませんでした。")

                if include_raw and result.raw_response is not None:
                    with st.expander("Raw response"):
                        st.json(result.raw_response)

                st.success("Lookup 完了")
