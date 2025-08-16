"""
llm_utils.py — Shared utilities for LLM prompt-based validation

Contains:
  - ValidationResult dataclass
  - LLMClient protocol
  - Prompt construction
  - Response parsing/coercion
  - CSV I/O helpers
  - Retry + batch orchestration
  - Heuristic (dry-run) client
  - OpenAI client (optionally attempts web search via Responses API)
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from dotenv import load_dotenv  # type: ignore
from tqdm.auto import tqdm  # type: ignore

# Public types
Parsed = Dict[str, Any]


@dataclass
class ValidationResult:
    """Structured output for one validated record."""

    input: Dict[str, Any]
    output: Dict[str, Any]
    raw_response: str
    attempts: int


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:  # pragma: no cover - interface only
        ...


# -----------------------------
# Prompt Construction
# -----------------------------

def build_validation_prompt(record: Dict[str, Any]) -> Tuple[str, str]:
    """Build system and user prompts for LLM-based venue validation (Sections III–IV)."""
    system_prompt = (
        "You are a data quality validator for venue records in a B2B onboarding pipeline. "
        "Decide whether a venue record is VALID or INVALID for onboarding based only on the structured input. "
        "A venue is VALID if it appears actionable: appropriate business category, likely active, and minimally verifiable. "
        "A venue is INVALID if it is closed, duplicative, miscategorized, lacks evidence of being a real venue, or otherwise non-actionable. "
        "Return ONLY a JSON object strictly following the schema provided, with no additional commentary."
    )

    def _fmt_bool(x: Any) -> str:
        if x is None:
            return "null"
        if isinstance(x, bool):
            return "true" if x else "false"
        s = str(x).strip().lower()
        return "true" if s in {"true", "1", "yes", "y"} else "false"

    user_prompt = (
        f"- business_name: {record.get('business_name')}\n"
        f"- address: {record.get('address')}\n"
        f"- city: {record.get('city')}\n"
        f"- business_category: {record.get('business_category')}\n"
        f"- has_website: {_fmt_bool(record.get('has_website'))}\n"
        f"- address_present: {_fmt_bool(record.get('address_present'))}\n"
        f"- confidence_score: {record.get('confidence_score')}\n\n"
        "Output JSON only, exactly this schema:\n"
        "{\n"
        '  "is_valid": true | false,\n'
        '  "confidence": number,  // 0-100\n'
        '  "reason": string\n'
        "}"
    )
    return system_prompt, user_prompt


# -----------------------------
# Parsing
# -----------------------------

def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    return text[start : end + 1]


def parse_llm_response(raw_text: str) -> Parsed:
    """Parse LLM response into {is_valid, confidence (0-100), reason}."""
    try:
        candidate = _extract_first_json_object(raw_text)
        candidate = candidate.replace("\u00A0", " ")
        candidate = re.sub(r"\bTrue\b", "true", candidate)
        candidate = re.sub(r"\bFalse\b", "false", candidate)
        candidate = re.sub(r"\bNone\b", "null", candidate)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        if candidate.count("'") > candidate.count('"'):
            candidate = candidate.replace("'", '"')

        obj = json.loads(candidate)
        is_valid = bool(obj.get("is_valid", False))
        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(100.0, confidence))
        reason = str(obj.get("reason", ""))[:500]
        return {"is_valid": is_valid, "confidence": confidence, "reason": reason}
    except Exception:
        return {"is_valid": False, "confidence": 0.0, "reason": "manual_review"}


# -----------------------------
# CSV I/O Utilities
# -----------------------------

def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_records_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record: Dict[str, Any] = {
                "business_name": row.get("business_name") or row.get("name") or "",
                "address": row.get("address") or None,
                "city": row.get("city") or "",
                "business_category": row.get("business_category") or row.get("category") or "",
                "has_website": _to_bool(row.get("has_website")),
                "address_present": _to_bool(row.get("address_present")),
                "confidence_score": _to_float(row.get("confidence_score")),
            }
            records.append(record)
    return records


def write_results_to_jsonl(results: List[ValidationResult], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "input": r.input,
                "output": r.output,
                "raw_response": r.raw_response,
                "attempts": r.attempts,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# Retry + Batch orchestration
# -----------------------------

def validate_record_with_retries(
    record: Dict[str, Any],
    llm: LLMClient,
    max_retries: int = 2,
    initial_backoff_s: float = 0.5,
) -> ValidationResult:
    system_prompt, user_prompt = build_validation_prompt(record)
    attempt = 0
    while True:
        try:
            response_text = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = parse_llm_response(response_text)
            return ValidationResult(
                input=record,
                output=parsed,
                raw_response=response_text,
                attempts=attempt + 1,
            )
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                fallback = parse_llm_response("")
                fallback_reason = fallback.get("reason", "manual_review")
                fallback["reason"] = f"{fallback_reason}; runtime_error: {str(e)}"
                return ValidationResult(input=record, output=fallback, raw_response="", attempts=attempt)
            backoff = initial_backoff_s * (2 ** (attempt - 1))
            jitter = 0.5 + random.random()  # 0.5x–1.5x
            time.sleep(backoff * jitter)


def validate_batch(
    records: Iterable[Dict[str, Any]],
    llm: LLMClient,
    max_workers: int = 4,
    max_retries: int = 2,
    initial_backoff_s: float = 0.5,
) -> List[ValidationResult]:
    records_list = list(records)
    results: List[ValidationResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                validate_record_with_retries,
                rec,
                llm,
                max_retries,
                initial_backoff_s,
            )
            for rec in records_list
        ]
        with tqdm(total=len(futures), desc="Validating", unit="rec") as pbar:
            for f in as_completed(futures):
                results.append(f.result())
                pbar.update(1)
    return results


# -----------------------------
# LLM Clients
# -----------------------------
class RuleBasedHeuristicLLM:
    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:  # pragma: no cover - simple heuristic
        def _extract(field: str) -> Optional[str]:
            pattern = rf"- {re.escape(field)}: (.*)\n"
            m = re.search(pattern, user_prompt)
            return m.group(1).strip() if m else None

        has_website_str = (_extract("has_website") or "false").lower()
        address_present_str = (_extract("address_present") or "false").lower()
        conf_score_str = _extract("confidence_score") or "0"
        category_str = (_extract("business_category") or "").lower()

        has_website = has_website_str in {"true", "1", "yes"}
        address_present = address_present_str in {"true", "1", "yes"}
        try:
            conf_score = float(conf_score_str)
        except Exception:
            conf_score = 0.0

        likely_valid = has_website or address_present or conf_score >= 0.6
        miscategorized = any(tok in category_str for tok in ["unknown", "misc", "other"]) and not likely_valid

        is_valid = bool(likely_valid and not miscategorized)
        confidence = 80.0 if is_valid else 30.0
        reason = (
            ("has_website; " if has_website else "no_website; ")
            + ("address_present; " if address_present else "address_missing; ")
            + f"confidence_score={conf_score:.2f}"
        )

        return json.dumps({"is_valid": is_valid, "confidence": confidence, "reason": reason})


class OpenAIChatClient:
    """OpenAI client.

    By default uses Chat Completions. If use_web_search=True, attempts Responses API
    with web search tool (requires compatible SDK/version and feature availability).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 256,
        use_web_search: bool = False,
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Add it to your environment or a .env file.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("The 'openai' package is required. Install with: pip install openai") from e

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.use_web_search = bool(use_web_search)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))

        if self.use_web_search:
            try:
                resp = self._client.responses.create(  # type: ignore[attr-defined]
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=[{"type": "web_search"}],
                )
                content = resp.output_text
                if content:
                    return content
            except Exception:
                # Gracefully fall back to Chat Completions if web search is unsupported
                pass

        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
        return content or "" 