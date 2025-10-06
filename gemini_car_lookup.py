"""Service for looking up vehicle specs by kata-shiki (model codes) using Gemini."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

__all__ = [
    "GeminiCarLookupError",
    "VehicleMatch",
    "LookupResult",
    "GeminiCarLookupService",
]

GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
PROMPT_TEMPLATE = (
    "あなたは自動車データリサーチアシスタントです。"
    "車両型式『{model_code}』に対応する車両情報を、Google検索ツールで調査してください"
    "{language_instruction}"
    "LLMの事前知識や学習データに基づく推測・補完は一切行わず、"
    "必ず実在する情報源に裏付けられたデータのみを使用してください。"
    "裏付けの取れない情報は回答に含めてはいけません。"
    "複数の車種が確認された場合は、別オブジェクトとして扱ってください。"
    "回答はJSON形式のみとし、トップレベルに`vehicles`配列を置きます。"
    "各要素には manufacturer, vehicle_name, displacement_cc（整数）, "
    "confidence（'high'|'medium'|'low'）, grade_or_variant(グレード配列), sources（URL配列）を含めてください。"
    "根拠が弱い場合でも候補を返してよいが、confidence を 'medium' または 'low' に設定し、"
    "全く根拠が得られない場合のみ、`vehicles` を空配列とし、notes にその理由を説明します。"
    "情報の捏造・推測・補完は禁止です。"
    "Never guess, infer, or hallucinate likely models."
    "Perform at most one grounded web search query, and stop after the first valid source is found."
)

DEFAULT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "vehicles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "manufacturer": {"type": "STRING"},
                    "vehicle_name": {"type": "STRING"},
                    "displacement_cc": {"type": "NUMBER"},
                    "grade_or_variant": {"type": "STRING"},
                    "years": {"type": "STRING"},
                    "confidence": {"type": "STRING"},
                    "sources": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": [
                    "manufacturer",
                    "vehicle_name",
                    "displacement_cc",
                    "confidence",
                    "sources",
                ],
            },
        }
    },
    "required": ["vehicles"],
}


def _load_env_from_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


_load_env_from_file()


def _strip_code_fence(text: str) -> str:
    """Remove Markdown-style code fences if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    parts = stripped.split("```")
    if len(parts) < 2:
        return stripped

    inner = parts[1]
    if inner.startswith("json"):
        inner = inner[len("json"):]
    return inner.strip()


class GeminiCarLookupError(RuntimeError):
    """Raised when Gemini response parsing or retrieval fails."""


@dataclass
class VehicleMatch:
    manufacturer: Optional[str]
    vehicle_name: Optional[str]
    displacement_cc: Optional[int]
    grade_or_variant: Optional[str] = None
    years: Optional[str] = None
    confidence: Optional[str] = None
    notes: Optional[str] = None
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation free of Nones."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v not in (None, [], "")}


@dataclass
class LookupResult:
    model_code: str
    matches: List[VehicleMatch]
    raw_response: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_code": self.model_code,
            "matches": [match.to_dict() for match in self.matches],
            **({"usage_metadata": self.usage_metadata} if self.usage_metadata is not None else {}),
            **({"raw_response": self.raw_response} if self.raw_response is not None else {}),
        }


class GeminiCarLookupService:
    """Wrapper around Gemini's grounded search to resolve model codes into specs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.5-flash",
        timeout_seconds: int = 30,
        include_raw_response: bool = False,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_language: str = "日本語",
        response_mime_type: Optional[str] = "auto",
        response_schema: Optional[Dict[str, Any]] = None,
        allow_google_search_fallback: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided. Pass api_key or set environment variable.")

        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.include_raw_response = include_raw_response
        self.endpoint = GEMINI_API_URL_TEMPLATE.format(model=model_name)
        if tools is not None:
            self.tools = list(tools)
        else:
            if "2.5" in self.model_name:
                # Gemini auto-selects grounding strength; no explicit mode field is accepted now.
                self.tools = [{"googleSearchRetrieval": {}}]
            else:
                self.tools = [{"googleSearch": {}}]
        self._uses_search_retrieval = any(
            isinstance(tool, dict) and "googleSearchRetrieval" in tool for tool in self.tools
        )
        self.allow_google_search_fallback = allow_google_search_fallback
        self.system_instruction = system_instruction or (
            "回答は必ず信頼性の高い自動車情報源に基づくこと。"
            "広告的・推測的な情報や、出典間で内容が食い違うものは使用しないこと。"
        )
        self.response_language = response_language
        if response_mime_type == "auto":
            self.response_mime_type: Optional[str] = None if "2.5" in self.model_name else "application/json"
        else:
            self.response_mime_type = response_mime_type
        if response_schema is None and "2.5" in self.model_name:
            self.response_schema = DEFAULT_RESPONSE_SCHEMA
        else:
            self.response_schema = response_schema

    def lookup(self, model_code: str) -> LookupResult:
        if not model_code or not model_code.strip():
            raise ValueError("model_code must be a non-empty string")

        normalized_code = model_code.strip()
        payload = self._build_payload(normalized_code)
        response = self._call_api(payload)
        matches = self._parse_matches(response, normalized_code)
        return LookupResult(
            model_code=normalized_code,
            matches=matches,
            raw_response=response if self.include_raw_response else None,
            usage_metadata=response.get("usageMetadata"),
        )

    def _build_payload(self, model_code: str) -> Dict[str, Any]:
        language_instruction = (
            f"\n全ての文字列フィールド(manufacturer, vehicle_name, grade_or_variant, years, notes, confidence)は{self.response_language}で記述してください。"
            if self.response_language
            else ""
        )
        user_prompt = PROMPT_TEMPLATE.format(
            model_code=model_code,
            language_instruction=language_instruction,
        )
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "tools": self.tools,
        }
        generation_config: Dict[str, Any] = {
            "temperature": 0.0,
            "topK": 1,
            "topP": 1.0,
        }
        if self.response_schema:
            generation_config["responseSchema"] = self.response_schema
        elif self.response_mime_type:
            generation_config["responseMimeType"] = self.response_mime_type
        payload["generationConfig"] = generation_config
        if self.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": self.system_instruction}]}
        return payload

    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = {"key": self.api_key}
        response = self._perform_request(payload, params)

        if response.status_code == 200:
            return self._decode_response_json(response)

        if self._should_retry_without_grounding(response.status_code, response.text):
            if self.allow_google_search_fallback:
                fallback_payload, fallback_tools = self._build_google_search_fallback(payload)
                fallback_response = self._perform_request(fallback_payload, params)
                if fallback_response.status_code == 200:
                    self.tools = fallback_tools
                    self._uses_search_retrieval = any(
                        isinstance(tool, dict) and "googleSearchRetrieval" in tool for tool in self.tools
                    )
                    return self._decode_response_json(fallback_response)
                raise GeminiCarLookupError(
                    f"Gemini API error {fallback_response.status_code}: {fallback_response.text[:200]}"
                )
            raise GeminiCarLookupError(
                "Search Grounding is not supported for this API key/project. "
                "Enable Search Grounding or instantiate GeminiCarLookupService with "
                "allow_google_search_fallback=True to fall back to the legacy googleSearch tool."
            )

        raise GeminiCarLookupError(
            f"Gemini API error {response.status_code}: {response.text[:200]}"
        )

    def _perform_request(self, payload: Dict[str, Any], params: Dict[str, Any]) -> requests.Response:
        try:
            return requests.post(
                self.endpoint,
                params=params,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise GeminiCarLookupError(f"HTTP request to Gemini failed: {exc}") from exc

    def _decode_response_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise GeminiCarLookupError("Failed to decode JSON from Gemini response") from exc

    def _should_retry_without_grounding(self, status_code: int, response_body: str) -> bool:
        if status_code != 400 or not self._uses_search_retrieval:
            return False
        lowered = (response_body or "").lower()
        return "search grounding is not supported" in lowered or "groundingmode" in lowered

    def _build_google_search_fallback(
        self, payload: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        sanitized = copy.deepcopy(payload)
        tools = sanitized.get("tools", []) or []
        fallback_tools: List[Dict[str, Any]] = []
        has_google_search = False
        for tool in tools:
            if isinstance(tool, dict):
                if "googleSearchRetrieval" in tool:
                    continue
                if "googleSearch" in tool:
                    has_google_search = True
            fallback_tools.append(tool)
        if not has_google_search:
            fallback_tools.append({"googleSearch": {}})
        sanitized["tools"] = fallback_tools
        return sanitized, fallback_tools

    def _parse_matches(self, response: Dict[str, Any], model_code: str) -> List[VehicleMatch]:
        candidates = response.get("candidates") or []
        if not candidates:
            raise GeminiCarLookupError("Gemini returned no candidates")

        primary = candidates[0]
        structured = self._extract_structured_candidate(primary)
        if structured is None:
            response_text = self._collect_text_parts(primary)
            if not response_text:
                raise GeminiCarLookupError("Gemini candidate missing text content")

            structured = self._load_structured_payload(response_text)
        vehicle_payloads = self._extract_vehicle_payloads(structured)
        sources = self._extract_grounded_sources(primary)

        matches: List[VehicleMatch] = []
        for item in vehicle_payloads:
            matches.append(self._build_match(item, sources))

        matches = self._expand_vehicle_names(matches)

        if not matches:
            matches.append(
                VehicleMatch(
                    manufacturer=None,
                    vehicle_name=None,
                    displacement_cc=None,
                    notes=f"No grounded data found for model code '{model_code}'.",
                    confidence="low",
                    sources=sources,
                )
            )
        return matches

    @staticmethod
    def _collect_text_parts(candidate: Dict[str, Any]) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(filter(None, text_parts)).strip()

    @staticmethod
    def _extract_structured_candidate(candidate: Dict[str, Any]) -> Optional[Any]:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "structValue" in part:
                return GeminiCarLookupService._convert_struct_value(part["structValue"])
            if "jsonValue" in part:
                return part["jsonValue"]
        return None

    @staticmethod
    def _convert_struct_value(struct_value: Dict[str, Any]) -> Dict[str, Any]:
        fields = struct_value.get("fields")
        if not isinstance(fields, dict):
            return struct_value  # unexpected shape; return raw
        return {
            key: GeminiCarLookupService._convert_proto_value(value)
            for key, value in fields.items()
        }

    @staticmethod
    def _convert_proto_value(value: Any) -> Any:
        if isinstance(value, dict):
            if "stringValue" in value:
                return value["stringValue"]
            if "numberValue" in value:
                return value["numberValue"]
            if "boolValue" in value:
                return value["boolValue"]
            if "nullValue" in value:
                return None
            if "structValue" in value:
                return GeminiCarLookupService._convert_struct_value(value["structValue"])
            if "listValue" in value:
                list_payload = value["listValue"].get("values", [])
                return [GeminiCarLookupService._convert_proto_value(item) for item in list_payload]
            if "jsonValue" in value:
                return value["jsonValue"]
        return value

    @staticmethod
    def _load_structured_payload(text: str) -> Any:
        cleaned = _strip_code_fence(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            try:
                obj, index = decoder.raw_decode(cleaned)
            except json.JSONDecodeError:
                raise GeminiCarLookupError(
                    "Gemini response was not valid JSON. Enable include_raw_response for troubleshooting."
                ) from exc

            # Allow Gemini to append explanatory text after the JSON block.
            trailing = cleaned[index:].strip()
            if trailing:
                # Future: log trailing text when raw_response is requested.
                pass
            return obj

    @staticmethod
    def _extract_vehicle_payloads(structured: Any) -> List[Dict[str, Any]]:
        if isinstance(structured, dict):
            if "vehicles" in structured and isinstance(structured["vehicles"], list):
                return [item for item in structured["vehicles"] if isinstance(item, dict)]
            return [structured]
        if isinstance(structured, list):
            return [item for item in structured if isinstance(item, dict)]
        raise GeminiCarLookupError("Unexpected JSON structure; expected object with vehicles array.")

    @staticmethod
    def _extract_grounded_sources(candidate: Dict[str, Any]) -> List[str]:
        sources: List[str] = []
        metadata = candidate.get("groundingMetadata") or {}

        # Extract URLs from grounding chunks when present.
        for chunk in metadata.get("groundingChunks", []):
            if isinstance(chunk, dict):
                web = chunk.get("web") or {}
                url = web.get("uri") or web.get("url")
                if url:
                    sources.append(url)

        # Some responses contain grounding supports referencing chunk indices.
        for support in metadata.get("groundingSupports", []):
            if not isinstance(support, dict):
                continue
            for ref in support.get("groundingChunks", []):
                if not isinstance(ref, dict):
                    continue
                web = ref.get("web") or {}
                url = web.get("uri") or web.get("url")
                if url:
                    sources.append(url)

        # Fallback to top-level citations list if exposed.
        for citation in metadata.get("citations", []):
            if isinstance(citation, dict):
                url = citation.get("url")
                if url:
                    sources.append(url)

        # Remove duplicates while preserving order.
        deduped: Dict[str, None] = {}
        for url in sources:
            if url not in deduped:
                deduped[url] = None
        return list(deduped.keys())

    @staticmethod
    def _build_match(item: Dict[str, Any], default_sources: Iterable[str]) -> VehicleMatch:
        displacement = item.get("displacement_cc")
        if isinstance(displacement, str) and displacement.isdigit():
            displacement = int(displacement)
        elif isinstance(displacement, (int, float)):
            displacement = int(displacement)
        else:
            displacement = None

        sources = list(default_sources)
        extra_sources = item.get("sources")
        if isinstance(extra_sources, (list, tuple)):
            for url in extra_sources:
                if isinstance(url, str) and url not in sources:
                    sources.append(url)

        confidence = item.get("confidence")
        if confidence and isinstance(confidence, str):
            confidence = confidence.lower()

        return VehicleMatch(
            manufacturer=item.get("manufacturer"),
            vehicle_name=item.get("vehicle_name") or item.get("car_name") or item.get("model"),
            displacement_cc=displacement,
            grade_or_variant=item.get("grade_or_variant"),
            years=item.get("years"),
            confidence=confidence,
            notes=item.get("notes"),
            sources=sources,
        )

    @staticmethod
    def _split_vehicle_name(name: str) -> List[str]:
        if not name:
            return []
        for delimiter in ("/", "／"):
            if delimiter in name:
                parts = [part.strip() for part in name.split(delimiter) if part.strip()]
                if len(parts) > 1:
                    return parts
        return [name]

    @staticmethod
    def _expand_vehicle_names(matches: List[VehicleMatch]) -> List[VehicleMatch]:
        expanded: List[VehicleMatch] = []
        for match in matches:
            name = (match.vehicle_name or "").strip()
            if not name:
                expanded.append(match)
                continue

            split_names = GeminiCarLookupService._split_vehicle_name(name)
            if len(split_names) <= 1:
                expanded.append(match)
                continue

            for split_name in split_names:
                expanded.append(replace(match, vehicle_name=split_name))

        return expanded


def lookup_car(model_code: str, **kwargs: Any) -> LookupResult:
    """Convenience function for single-shot lookups."""
    service = GeminiCarLookupService(**kwargs)
    return service.lookup(model_code)
