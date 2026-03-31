from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from urllib import error, request

from resupply_engine.models import (
    CanonicalCase,
    CompiledClinicalWorkbook,
    LLMRecommendationResponse,
    SupplyCatalogItem,
)


class LLMRecommendationError(RuntimeError):
    pass


class BaseRecommendationBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaBackend:
    """Local Ollama backend that calls the on-device Ollama REST API."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434/api",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.0,
        max_tokens: int = 600,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode("utf-8")
        request_url = f"{self.base_url}/generate"
        http_request = request.Request(
            request_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise LLMRecommendationError(
                f"Ollama returned HTTP {exc.code} for model {self.model}: {details}"
            ) from exc
        except error.URLError as exc:
            raise LLMRecommendationError(
                f"Could not reach Ollama at {request_url}. Is `ollama serve` running?"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive around local runtime
            raise LLMRecommendationError(f"Ollama inference failed: {exc}") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise LLMRecommendationError("Ollama returned invalid JSON.") from exc

        generated_text = decoded.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LLMRecommendationError("Ollama returned no generated response text.")
        return generated_text.strip()


class LocalHuggingFaceBackend:
    """Optional local Hugging Face backend for offline supply recommendation."""

    def __init__(
        self,
        model_path: Path,
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
            raise LLMRecommendationError(
                "transformers is not installed for local Hugging Face inference."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            model = AutoModelForCausalLM.from_pretrained(str(self.model_path))
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            output = generator(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model availability
            raise LLMRecommendationError(f"Local Hugging Face inference failed: {exc}") from exc

        if not output:
            raise LLMRecommendationError("Local Hugging Face backend returned no output.")
        return str(output[0].get("generated_text", "")).strip()


class FakeJSONBackend:
    """Test helper backend that returns a fixed JSON string."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, prompt: str) -> str:
        return self.response_text


class LocalLLMRecommender:
    def __init__(self, backend: BaseRecommendationBackend | None = None) -> None:
        self.backend = backend

    def recommend(
        self,
        canonical_case: CanonicalCase,
        workbook: CompiledClinicalWorkbook,
        catalog: dict[str, SupplyCatalogItem],
    ) -> LLMRecommendationResponse:
        if self.backend is None:
            raise LLMRecommendationError("No local LLM backend configured.")

        prompt = build_llm_prompt(canonical_case, workbook, catalog)
        raw_response = self.backend.generate(prompt)
        json_payload = _extract_json_object(raw_response)
        try:
            return LLMRecommendationResponse.model_validate_json(json_payload)
        except Exception as exc:
            raise LLMRecommendationError(f"LLM returned invalid structured JSON: {exc}") from exc


def build_llm_prompt(
    canonical_case: CanonicalCase,
    workbook: CompiledClinicalWorkbook,
    catalog: dict[str, SupplyCatalogItem],
) -> str:
    catalog_payload = [
        {
            "item_id": item.item_id,
            "name": item.name,
            "unit_weight_lb": item.unit_weight_lb,
            "notes": item.notes,
        }
        for item in sorted(catalog.values(), key=lambda item: (item.name, item.item_id))
    ]
    workbook_payload = [
        {
            "rule_id": rule.rule_id,
            "item_id": rule.item_id,
            "logic_type": rule.logic_type,
            "trigger_text": rule.trigger_text,
            "clinical_rationale": rule.clinical_rationale,
            "priority_rank": rule.priority_rank,
            "unsupported_clauses": rule.unsupported_clauses,
        }
        for rule in workbook.item_rules
    ]
    prompt_payload = {
        "canonical_case": canonical_case.model_dump(mode="json"),
        "catalog": catalog_payload,
        "clinical_workbook_rules": workbook_payload,
        "priority_ranking": {
            item_id: rank.model_dump(mode="json")
            for item_id, rank in workbook.priority_ranks.items()
        },
        "instructions": {
            "task": ["You are a medical supply recommendation system, specifically designed to recommend what supplies to send for casualties with traumatic brain injuries. "
            "Analyze the provided case and recommend the appropriate supplies."],
            
            "constraints": [
                "Return only JSON.",
                "Only recommend item_id values that appear in the catalog.",
                "Do not recommend manifests, drones, packing, or redundancy.",
                "Use positive integer quantities only.",
                "Do not include markdown fences or explanation outside the JSON object.",
            ],
            "response_schema": {
                "recommended_items": [
                    {
                        "item_id": "string",
                        "quantity": 1,
                        "rationale": "string",
                        "policy_refs": ["string"],
                        "confidence": 0.0,
                    }
                ],
                "overall_confidence": 0.0,
                "manual_review_notes": "string",
            },
        },
    }
    return json.dumps(prompt_payload, indent=2, sort_keys=True)


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMRecommendationError("LLM response did not contain a JSON object.")
    return text[start : end + 1]
