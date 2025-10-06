"""Service for looking up vehicle specs by kata-shiki (model codes) using Gemini."""

from __future__ import annotations

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
    "You are an automotive research assistant. Given a Japanese vehicle model code (kata-shiki), "
    "use grounded web search to locate authoritative sources. {language_instruction} "
    "Provide a separate object for each distinct vehicle or market name; do not merge multiple names into a single entry. "
    "Respond strictly as JSON with a top-level object containing a `vehicles` array. "
    "Each entry must include: manufacturer, vehicle_name, displacement_cc (integer), confidence "
    "('high'|'medium'|'low'), optional grade_or_variant"
    "If the lookup fails, return an empty `vehicles` array. "
    "Never invent data that is not grounded in the cited sources. Model code: {model_code}."
)


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
    confidence: Optional[str] = None

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


def _normalize_model_name(model_name: str) -> str:
    """Ensure model identifiers include the `models/` prefix."""
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    if model_name.startswith(("models/", "tunedModels/")):
        return model_name
    return f"models/{model_name}"


class GeminiCarLookupService:
    """Wrapper around Gemini's grounded search to resolve model codes into specs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        timeout_seconds: int = 30,
        include_raw_response: bool = False,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_language: str = "日本語",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided. Pass api_key or set environment variable.")

        self.model_name = _normalize_model_name(model_name)
        self.timeout_seconds = timeout_seconds
        self.include_raw_response = include_raw_response
        self.endpoint = GEMINI_API_URL_TEMPLATE.format(model=self.model_name)
        self.tools = list(tools) if tools is not None else [{"googleSearch": {}}]
        self.system_instruction = system_instruction or (
            "Ground every answer in reputable automotive sources (OEM documentation, trusted press, "
            "dealership listings). Refuse to answer if sources conflict or cannot be verified."
        )
        self.response_language = response_language

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
            "Return all textual field values"
            " (manufacturer, vehicle_name, grade_or_variant, confidence) "
            f"in {self.response_language}. "
            f"If source titles are quoted, localise descriptive text while keeping URL domains intact. "
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
            "generationConfig": {
                "temperature": 0.0,
                "topK": 1,
                "topP": 1.0,
            },
        }
        if self.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": self.system_instruction}]}
        return payload

    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = {"key": self.api_key}
        try:
            resp = requests.post(
                self.endpoint,
                params=params,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise GeminiCarLookupError(f"HTTP request to Gemini failed: {exc}") from exc

        if resp.status_code != 200:
            raise GeminiCarLookupError(
                f"Gemini API error {resp.status_code}: {resp.text[:200]}"
            )

        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise GeminiCarLookupError("Failed to decode JSON from Gemini response") from exc

    def _parse_matches(self, response: Dict[str, Any], model_code: str) -> List[VehicleMatch]:
        candidates = response.get("candidates") or []
        if not candidates:
            raise GeminiCarLookupError("Gemini returned no candidates")

        primary = candidates[0]
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
            confidence=confidence,
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
