from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from deterministic_extract import extract_mls_scalars
from document_ingest import anthropic_content, ingest_pdf
from property_schema import PropertyRecord


PROMPT_PATH = Path(__file__).parent / "prompts" / "property_extraction_v1.txt"


def _is_fact(value: Any) -> bool:
    return isinstance(value, dict) and "status" in value and (
        "value" in value or "evidence" in value
    )


def _merge_candidate(authoritative: Any, candidate: Any, path: str = "") -> Any:
    """
    Deterministic known values win. LLM values fill unknown fields. Material
    disagreements become unresolved conflicts instead of silent overwrites.
    """
    if authoritative is None:
        return deepcopy(candidate)
    if candidate is None:
        return deepcopy(authoritative)

    if _is_fact(authoritative) and _is_fact(candidate):
        a_known = authoritative.get("status") == "known"
        c_known = candidate.get("status") == "known"

        if a_known and c_known and authoritative.get("value") != candidate.get("value"):
            return {
                "status": "conflicted",
                "value": None,
                "evidence": (
                    authoritative.get("evidence", [])
                    + candidate.get("evidence", [])
                ),
                "notes": f"Conflicting candidates for {path}",
            }
        if a_known:
            return deepcopy(authoritative)
        if c_known:
            return deepcopy(candidate)
        return deepcopy(authoritative)

    if isinstance(authoritative, dict) and isinstance(candidate, dict):
        keys = authoritative.keys() | candidate.keys()
        return {
            key: _merge_candidate(
                authoritative.get(key),
                candidate.get(key),
                f"{path}.{key}".strip("."),
            )
            for key in keys
        }

    if isinstance(authoritative, list) and isinstance(candidate, list):
        # Evidence-bearing lists are retained. Deduplication can be made
        # field-specific later; losing evidence is worse than duplication.
        return deepcopy(authoritative) + deepcopy(candidate)

    return deepcopy(authoritative)


def _llm_candidate(
    mls_document,
    disclosure_document,
    model: str,
) -> dict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = []
    content.extend(anthropic_content(mls_document))
    content.extend(anthropic_content(disclosure_document))

    response = client.messages.create(
        model=model,
        max_tokens=12000,
        temperature=0,
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        messages=[{"role": "user", "content": content}],
        tools=[
            {
                "name": "record_property_evidence",
                "description": "Record document-supported property facts.",
                "input_schema": PropertyRecord.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "record_property_evidence"},
    )

    calls = [
        block
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
        and getattr(block, "name", None) == "record_property_evidence"
    ]
    if len(calls) != 1:
        raise RuntimeError(f"Expected one extraction tool call; received {len(calls)}")
    return calls[0].input


def build_property_record(
    mls_path: str,
    disclosure_path: str,
    model: str,
) -> PropertyRecord:
    mls = ingest_pdf(mls_path, kind="mls_listing")
    disclosure = ingest_pdf(disclosure_path, kind="property_disclosure")

    deterministic = extract_mls_scalars(mls)
    llm = _llm_candidate(mls, disclosure, model=model)
    merged = _merge_candidate(deterministic, llm)

    record = PropertyRecord.model_validate(merged)

    # The separate training-compatible remarks classifier owns this field.
    record.remarks_classification = None
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mls", required=True)
    parser.add_argument("--disclosure", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_MODEL"),
        help="Anthropic model ID; may also be set with ANTHROPIC_MODEL.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("Supply --model or set ANTHROPIC_MODEL")

    record = build_property_record(args.mls, args.disclosure, args.model)
    Path(args.output).write_text(
        record.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
