from gemini_car_lookup import lookup_car, GeminiCarLookupError

# Gemini 2.0 Flash (experimental) の暫定価格例。実際の料金は最新の公式表で確認してください。
PROMPT_RATE_PER_1K_TOKENS_USD = 0.00035
OUTPUT_RATE_PER_1K_TOKENS_USD = 0.0007


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

    print("Usage Metadata:", usage)
    print(
        f"Estimated cost: ${total_cost:.6f} "
        f"(prompt: ${prompt_cost:.6f}, output: ${output_cost:.6f})"
    )


def main() -> None:
    # TODO: 型式コードを必要に応じて差し替えてください。
    try:
        result = lookup_car("DBA-ZRR70W", include_raw_response=False)
    except GeminiCarLookupError as exc:
        print("Lookup failed:", exc)
        print("Tips: 一時的に include_raw_response=True を指定するとレスポンス全体を確認できます。")
        return

    for match in result.matches:
        print(match.to_dict())

    estimate_cost(result)


if __name__ == "__main__":
    main()
