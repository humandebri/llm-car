import argparse
import json

from gemini_car_lookup import lookup_car, GeminiCarLookupError

# Gemini 2.5 Flash の暫定価格: 1M トークンあたり $0.10。
# 最新の料金表を必ず確認してください。
PROMPT_RATE_PER_1K_TOKENS_USD = 0.0001
OUTPUT_RATE_PER_1K_TOKENS_USD = 0.0001
USD_TO_JPY_RATE = 150.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gemini lookup for a vehicle model code.",
    )
    parser.add_argument(
        "model_code",
        nargs="?",
        default="5BA-TALA15",
        help="Model code (kata-shiki) to look up.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Request and display the raw Gemini response payload.",
    )
    return parser.parse_args()


def estimate_cost(result) -> None:
    usage = result.usage_metadata or {}
    if not usage:
        print("Usage metadata was not returned; unable to estimate cost.")
        return

    prompt_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    prompt_cost = (prompt_tokens / 1000) * PROMPT_RATE_PER_1K_TOKENS_USD
    output_cost = (output_tokens / 1000) * OUTPUT_RATE_PER_1K_TOKENS_USD
    total_cost = prompt_cost + output_cost
    prompt_cost_jpy = prompt_cost * USD_TO_JPY_RATE
    output_cost_jpy = output_cost * USD_TO_JPY_RATE
    total_cost_jpy = total_cost * USD_TO_JPY_RATE

    print("Usage Metadata:", usage)
    print(
        f"Estimated cost: ${total_cost:.6f} (¥{total_cost_jpy:.2f}) "
        f"[prompt: ${prompt_cost:.6f} (¥{prompt_cost_jpy:.2f}), "
        f"output: ${output_cost:.6f} (¥{output_cost_jpy:.2f})]"
    )


def main() -> None:
    args = parse_args()

    # TODO: 型式コードを必要に応じて差し替えてください。
    try:
        result = lookup_car(
            args.model_code,
            include_raw_response=args.include_raw,
        )
    except GeminiCarLookupError as exc:
        print("Lookup failed:", exc)
        print("Tips: 一時的に include_raw_response=True を指定するとレスポンス全体を確認できます。")
        return

    for match in result.matches:
        print(match.to_dict())

    if args.include_raw:
        if result.raw_response is None:
            print("Raw response not available; ensure include_raw_response=True was set.")
        else:
            print("Raw response:")
            print(json.dumps(result.raw_response, ensure_ascii=False, indent=2))

    estimate_cost(result)


if __name__ == "__main__":
    main()
