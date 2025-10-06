# Gemini-grounded Car Lookup Service

## Goals
- Accept a JDM vehicle model code (型式) as input
- Query Gemini with Google Search grounding to obtain manufacturer, vehicle name, and displacement
- Return structured Python-friendly data (list of dicts) while exposing search citations for transparency

## High-level Flow
1. `GeminiCarLookupService.lookup(model_code)` normalizes the code and prepares a grounded prompt.
2. Service calls Gemini `generateContent` endpoint with the Google Search tool enabled and a response schema hint.
3. Gemini performs web search grounding, reasons over retrieved documents, and emits JSON describing one or more matching vehicles.
4. Service parses Gemini's JSON, extracts relevant fields, and pairs them with the cited URLs surfacing confidence & notes when uncertain.

## Components
- `GeminiCarLookupService`: orchestrates prompt crafting, API invocation, and result parsing.
- `VehicleMatch`: dataclass modeling a single grounded match (manufacturer, car name, displacement, notes, sources).
- `GroundedContentParser`: helper that extracts JSON payload & grounding metadata from Gemini responses and maps them onto `VehicleMatch` objects.

## Key Design Choices
- **Grounding-first prompt**: instruct Gemini to only answer when confident and to rely on cited sources to avoid hallucinations.
- **JSON contract**: prompts demand that Gemini replies with strict JSON that the parser can validate, reducing brittle parsing logic.
- **Source attribution**: grounding metadata is returned alongside each match so callers can audit provenance.
- **Pluggable model/tool config**: constructor accepts model name and optional tool overrides, easing experimentation (e.g., switching to 1.5-pro).

## Error Handling Strategy
- Empty or malformed responses raise `GeminiCarLookupError` with actionable context.
- Partial failures (e.g., missing displacement) surface via `notes` and `confidence` values instead of hard failures.
- Network / HTTP errors bubble up with descriptive messages to help operators decide on retries.

## Usage Sketch
```python
service = GeminiCarLookupService()
result = service.lookup("DBA-ZGE20G")
for match in result.matches:
    print(match.to_dict())
```
