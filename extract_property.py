"""
Extract a canonical PropertyRecord from an MLS PDF and a property-disclosure
PDF.

Claude's tool output is treated as untrusted candidate data. Candidate values
that conform to PropertyRecord are retained. Candidate values that do not
conform are discarded at the smallest practical semantic container, which then
falls back to the schema's normal unknown/default state. A warning is attached
to the returned record instead of allowing one optional value to fail the
entire extraction request.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from anthropic import Anthropic
from pydantic import ValidationError

from deterministic_extract import extract_mls_scalars
from document_ingest import anthropic_content, ingest_pdf
from property_schema import PropertyRecord


PROMPT_PATH = (
    Path(__file__).parent
    / "property_extraction_v1.txt"
)

MAX_VALIDATION_PASSES = 10
MAX_WARNING_VALUE_CHARS = 160


# ---------------------------------------------------------------------------
# MERGING
# ---------------------------------------------------------------------------

def _is_fact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "status" in value
        and (
            "value" in value
            or "evidence" in value
        )
    )


def _merge_candidate(
    authoritative: Any,
    candidate: Any,
    path: str = "",
) -> Any:
    """
    Merge deterministic and LLM-derived data.

    Deterministic known scalar facts win. LLM facts fill unknown fields.
    Material disagreements become unresolved Fact-level conflicts rather than
    silent overwrites. Lists are retained from both sources so evidence is not
    discarded during merging.
    """

    if authoritative is None:
        return deepcopy(candidate)

    if candidate is None:
        return deepcopy(authoritative)

    if _is_fact(authoritative) and _is_fact(candidate):
        authoritative_known = (
            authoritative.get("status") == "known"
        )
        candidate_known = (
            candidate.get("status") == "known"
        )

        if (
            authoritative_known
            and candidate_known
            and authoritative.get("value")
            != candidate.get("value")
        ):
            return {
                "status": "conflicted",
                "value": None,
                "evidence": (
                    authoritative.get("evidence", [])
                    + candidate.get("evidence", [])
                ),
                "notes": (
                    "Conflicting candidates for "
                    f"{path or 'unknown field'}"
                ),
            }

        if authoritative_known:
            return deepcopy(authoritative)

        if candidate_known:
            return deepcopy(candidate)

        return deepcopy(authoritative)

    if (
        isinstance(authoritative, dict)
        and isinstance(candidate, dict)
    ):
        keys = authoritative.keys() | candidate.keys()

        return {
            key: _merge_candidate(
                authoritative.get(key),
                candidate.get(key),
                f"{path}.{key}".strip("."),
            )
            for key in keys
        }

    if (
        isinstance(authoritative, list)
        and isinstance(candidate, list)
    ):
        return (
            deepcopy(authoritative)
            + deepcopy(candidate)
        )

    return deepcopy(authoritative)


# ---------------------------------------------------------------------------
# CLAUDE EXTRACTION
# ---------------------------------------------------------------------------

def _llm_candidate(
    mls_document,
    disclosure_document,
    model: str,
) -> dict:
    """
    Ask Claude to produce candidate data shaped like PropertyRecord.

    The returned dictionary is not trusted as a valid PropertyRecord until it
    has passed normalization and best-effort validation below.
    """

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    content: list[dict] = []
    content.extend(
        anthropic_content(mls_document)
    )
    content.extend(
        anthropic_content(disclosure_document)
    )

    response = client.messages.create(
        model=model,
        max_tokens=12000,
        temperature=0,
        system=PROMPT_PATH.read_text(
            encoding="utf-8"
        ),
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        tools=[
            {
                "name": "record_property_evidence",
                "description": (
                    "Record document-supported "
                    "property facts."
                ),
                "input_schema": (
                    PropertyRecord.model_json_schema()
                ),
            }
        ],
        tool_choice={
            "type": "tool",
            "name": "record_property_evidence",
        },
    )

    calls = [
        block
        for block in response.content
        if getattr(block, "type", None)
        == "tool_use"
        and getattr(block, "name", None)
        == "record_property_evidence"
    ]

    if len(calls) != 1:
        raise RuntimeError(
            "Expected exactly one property-extraction "
            f"tool call; received {len(calls)}"
        )

    candidate = calls[0].input

    if not isinstance(candidate, dict):
        raise RuntimeError(
            "Claude property-extraction tool output "
            "was not a JSON object"
        )

    return candidate


# ---------------------------------------------------------------------------
# SAFE, UNAMBIGUOUS NORMALIZATION
# ---------------------------------------------------------------------------

def _normalize_candidate(
    candidate: dict,
) -> dict:
    """
    Normalize only narrowly defined, semantically unambiguous differences.

    Normalization improves information retention. It is not required for
    pipeline survival: anything still invalid is later reset to unknown by
    _validate_best_effort().
    """

    normalized = deepcopy(candidate)

    # Normalize common basement label variants when the relevant structures
    # are dictionaries/lists. Unrecognized values are left untouched so the
    # validation layer can discard them transparently.
    basement = (
        normalized
        .get("basement_foundation", {})
        .get("basement_features", {})
    )

    if isinstance(basement, dict):
        basement_values = basement.get("values")

        if isinstance(basement_values, list):
            basement_aliases = {
                "unfinished basement": "unfinished",
                "walk-out": "walkout",
                "walk out": "walkout",
                "walk-out access": "walkout",
                "crawl": "crawl_space",
                "crawlspace": "crawl_space",
                "dirt": "dirt_floor",
            }

            normalized_values = []

            for value in basement_values:
                if isinstance(value, str):
                    cleaned = value.strip().lower()
                    normalized_values.append(
                        basement_aliases.get(
                            cleaned,
                            cleaned,
                        )
                    )
                else:
                    normalized_values.append(value)

            basement["values"] = normalized_values

    # If Claude places a year-only septic value in the complete-date field and
    # the schema provides septic_last_pumped_year, move it without inventing a
    # month or day.
    disclosures = normalized.get("disclosures")

    if isinstance(disclosures, dict):
        pumped_date = disclosures.get(
            "septic_last_pumped_date"
        )

        if (
            isinstance(pumped_date, dict)
            and pumped_date.get("status") == "known"
        ):
            value = pumped_date.get("value")
            year: int | None = None

            if (
                isinstance(value, str)
                and len(value.strip()) == 4
                and value.strip().isdigit()
            ):
                year = int(value.strip())

            elif (
                isinstance(value, int)
                and 1800 <= value <= 2100
            ):
                year = value

            if year is not None:
                year_fact = deepcopy(pumped_date)
                year_fact["value"] = year

                disclosures[
                    "septic_last_pumped_year"
                ] = year_fact

                disclosures.pop(
                    "septic_last_pumped_date",
                    None,
                )

    return normalized


# ---------------------------------------------------------------------------
# BEST-EFFORT VALIDATION
# ---------------------------------------------------------------------------

def _semantic_container_path(
    location: Iterable[Any],
) -> tuple[tuple[Any, ...], bool]:
    """
    Convert a Pydantic error location into the field to discard.

    Returns (path, reset_to_empty_model).

    For a Fact error such as disclosures.foo.value, the whole Fact is reset to
    {} so its Pydantic defaults become status=unknown and value=None.

    For an ObservedSet error such as interior.floors.values.0, the whole
    ObservedSet is reset to {} so its defaults become values=[] and
    complete=False.

    Other invalid fields are deleted so their declared schema defaults can be
    applied.
    """

    parts = tuple(location)

    for marker in ("values", "value"):
        if marker in parts:
            marker_index = parts.index(marker)
            container = parts[:marker_index]

            if container:
                return container, True

    return parts, False


def _navigate_to_parent(
    root: Any,
    path: tuple[Any, ...],
) -> tuple[Any, Any] | None:
    if not path:
        return None

    current = root

    for part in path[:-1]:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]

        elif (
            isinstance(current, list)
            and isinstance(part, int)
            and 0 <= part < len(current)
        ):
            current = current[part]

        else:
            return None

    return current, path[-1]


def _discard_invalid_path(
    root: dict,
    path: tuple[Any, ...],
    reset_to_empty_model: bool,
) -> bool:
    """Discard/reset one invalid path. Return True if data changed."""

    parent_and_key = _navigate_to_parent(
        root,
        path,
    )

    if parent_and_key is None:
        return False

    parent, key = parent_and_key

    if isinstance(parent, dict):
        if key not in parent:
            return False

        if reset_to_empty_model:
            if parent[key] == {}:
                return False
            parent[key] = {}
        else:
            del parent[key]

        return True

    if (
        isinstance(parent, list)
        and isinstance(key, int)
        and 0 <= key < len(parent)
    ):
        if reset_to_empty_model:
            if parent[key] == {}:
                return False
            parent[key] = {}
        else:
            parent.pop(key)

        return True

    return False


def _display_path(
    path: tuple[Any, ...],
) -> str:
    if not path:
        return "<record>"

    return ".".join(str(part) for part in path)


def _display_value(value: Any) -> str:
    rendered = repr(value)

    if len(rendered) > MAX_WARNING_VALUE_CHARS:
        rendered = (
            rendered[:MAX_WARNING_VALUE_CHARS]
            + "..."
        )

    return rendered


def _validate_best_effort(
    candidate: dict,
) -> PropertyRecord:
    """
    Return the most complete valid PropertyRecord that can be recovered.

    Validation errors are handled field by field. An invalid Fact or
    ObservedSet is reset to its unknown/default state. Other invalid fields are
    removed so their schema defaults apply. Each discarded value is reported
    in extraction_warnings.

    If repeated validation cannot recover a valid candidate, return a minimal
    valid PropertyRecord with warnings rather than raising ValidationError.
    """

    working = deepcopy(candidate)
    warnings: list[str] = []
    warning_keys: set[tuple[str, str, str]] = set()

    for _pass_number in range(
        1,
        MAX_VALIDATION_PASSES + 1,
    ):
        try:
            record = PropertyRecord.model_validate(
                working
            )

            for warning in warnings:
                if warning not in record.extraction_warnings:
                    record.extraction_warnings.append(
                        warning
                    )

            return record

        except ValidationError as validation_error:
            made_progress = False

            # Deeper/list paths are handled first. ObservedSet and Fact errors
            # usually collapse to shared semantic containers, so repeated
            # locations safely become no-ops after the first reset.
            problems = sorted(
                validation_error.errors(
                    include_url=False
                ),
                key=lambda problem: len(
                    problem.get("loc", ())
                ),
                reverse=True,
            )

            for problem in problems:
                location = tuple(
                    problem.get("loc", ())
                )

                path, reset_to_empty_model = (
                    _semantic_container_path(
                        location
                    )
                )

                if not path:
                    continue

                changed = _discard_invalid_path(
                    working,
                    path,
                    reset_to_empty_model,
                )

                if not changed:
                    continue

                made_progress = True

                path_text = _display_path(path)
                rejected_value = _display_value(
                    problem.get("input")
                )
                message = str(
                    problem.get(
                        "msg",
                        "Value did not conform to schema",
                    )
                )

                warning_key = (
                    path_text,
                    rejected_value,
                    message,
                )

                if warning_key not in warning_keys:
                    warning_keys.add(warning_key)
                    warnings.append(
                        "Discarded non-conforming extracted "
                        f"data at {path_text}; value="
                        f"{rejected_value}; reason={message}. "
                        "The field was left unknown."
                    )

            if not made_progress:
                warnings.append(
                    "Could not isolate all remaining schema "
                    "validation errors; returned a minimal "
                    "property record instead."
                )
                break

    else:
        warnings.append(
            "Reached the validation recovery limit; "
            "returned a minimal property record instead."
        )

    fallback = PropertyRecord()

    for warning in warnings:
        if warning not in fallback.extraction_warnings:
            fallback.extraction_warnings.append(
                warning
            )

    return fallback


# ---------------------------------------------------------------------------
# PUBLIC EXTRACTION FUNCTION
# ---------------------------------------------------------------------------

def build_property_record(
    mls_path: str,
    disclosure_path: str,
    model: str,
) -> PropertyRecord:
    """
    Build the best available canonical record from two property documents.

    PDF/Anthropic infrastructure failures still raise an exception. Invalid or
    incomplete individual property values do not: they become unknown fields
    accompanied by extraction warnings.
    """

    mls = ingest_pdf(
        mls_path,
        kind="mls_listing",
    )
    disclosure = ingest_pdf(
        disclosure_path,
        kind="property_disclosure",
    )

    deterministic = extract_mls_scalars(mls)

    llm = _llm_candidate(
        mls,
        disclosure,
        model=model,
    )

    llm = _normalize_candidate(llm)

    merged = _merge_candidate(
        deterministic,
        llm,
    )

    record = _validate_best_effort(merged)

    # Remarks classification is performed separately with the versioned,
    # training-compatible remarks prompts.
    record.remarks_classification = None

    return record


# ---------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mls",
        required=True,
    )
    parser.add_argument(
        "--disclosure",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_MODEL"),
        help=(
            "Anthropic model ID; may also be set "
            "with ANTHROPIC_MODEL."
        ),
    )

    args = parser.parse_args()

    if not args.model:
        parser.error(
            "Supply --model or set ANTHROPIC_MODEL"
        )

    record = build_property_record(
        args.mls,
        args.disclosure,
        args.model,
    )

    Path(args.output).write_text(
        record.model_dump_json(
            indent=2,
            exclude_none=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()