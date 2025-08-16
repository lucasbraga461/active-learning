"""
llm_validation.py — Runner for LLM Prompt-Based Validation (Sections III–IV)

Thin wrapper that selects a provider client, loads records, validates them in
batch using utilities from llm_utils.py, and writes results to JSONL.
"""
from __future__ import annotations

import argparse

from llm_utils import (
    OpenAIChatClient,
    RuleBasedHeuristicLLM,
    ValidationResult,
    load_records_from_csv,
    validate_batch,
    write_results_to_jsonl,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Prompt-Based Validation (runner)")
    p.add_argument("--input-csv", required=True, help="Path to input CSV")
    p.add_argument("--output-jsonl", required=True, help="Path to output JSONL with results")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--initial-backoff-s", type=float, default=0.5)

    # Provider selection
    p.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic (no external API)")
    p.add_argument("--provider", choices=["openai"], default=None, help="Online LLM provider to use")

    # OpenAI options
    # Default to a browsing-capable series; user can change as needed
    p.add_argument("--openai-model", type=str, default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--use-web-search", action="store_true", help="Attempt web search via OpenAI Responses API")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    records = load_records_from_csv(args.input_csv)

    if args.dry_run:
        llm = RuleBasedHeuristicLLM()
    else:
        if args.provider == "openai" or args.provider is None:
            llm = OpenAIChatClient(
                model=args.openai_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_web_search=args.use_web_search,
            )
        else:  # pragma: no cover - future providers
            raise RuntimeError(f"Unsupported provider: {args.provider}")

    results: list[ValidationResult] = validate_batch(
        records=records,
        llm=llm,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        initial_backoff_s=args.initial_backoff_s,
    )

    write_results_to_jsonl(results, args.output_jsonl)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# Example usage (terminal)
#
# OpenAI (reads OPENAI_API_KEY from environment or .env):
#    # .env at repo root: OPENAI_API_KEY=sk-...
#    python llm-prompt-based-validation/llm_validation.py \
#      --input-csv llm-prompt-based-validation/dummy_dataset.csv \
#      --output-jsonl llm-prompt-based-validation/results_openai.jsonl \
#      --provider openai \
#      --openai-model gpt-4o \
#      --use-web-search \
#      --temperature 0.2 \
#      --max-tokens 256
# ----------------------------------------------------------------------