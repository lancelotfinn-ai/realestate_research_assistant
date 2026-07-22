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
from property_schema import (
    ExtractionMethod,
    PropertyRecord,
    RemarksClassification,
    SourceType,
)


PROMPT_PATH = (
    Path(__file__).parent
    / "property_extraction_v1.txt"
)

MAX_VALIDATION_PASSES = 10
MAX_WARNING_VALUE_CHARS = 160

REMARKS_PROMPT_VERSION = "remarks-v1-deploy-2026-07"
REMARKS_MAX_CHARS = 24000

REMARKS_SYSTEM_PROMPT = """
You classify public MLS remarks into the supplied RemarksClassification
schema. The classifications must match the concepts used to train a Maine
hedonic home-price model.

Use only the supplied public remarks. Do not use the seller disclosure,
private remarks, general knowledge, or assumptions about the property.

Every scalar classification is a Fact object. For each known Fact include:
- status: "known"
- the schema-compatible value
- evidence containing source_type="mls_remarks",
  extraction_method="llm", a short exact excerpt or close paraphrase, and a
  confidence from 0 to 1

Use status="unknown" with value=null when the text does not support a
classification, except where the rules below explicitly define an absent
marketing signal as false or "none". Do not invent renovations, defects,
views, privacy, or seller motivation.

CLASSIFICATION RULES

condition:
- move-in-ready: turn-key, renovated throughout, nothing needed, or built
  since 2019 with no damage or repair concern stated
- updated: meaningful updates are mentioned, but not a whole-home renovation
- dated: original/vintage finishes or positive quality language without
  renovation/system-update evidence
- needs-work: repairs or updates are explicitly needed
- fixer: fixer-upper, project, TLC, or priced for condition
- unknown: insufficient condition information

new_roof, new_heating, new_windows, new_basement_work:
- true only when the corresponding recent replacement or improvement is
  explicitly stated
- false only when the remarks explicitly contradict the claim
- otherwise unknown

systems_updated:
- true when heating, plumbing, electrical, roof, or comparable major systems
  are described as new or recently replaced
- false only when explicitly described as original/not updated
- otherwise unknown

water_issues:
- true for explicit water intrusion, wet basement, flooding, or moisture
- false only for an explicit statement that the basement/property is dry or
  free of water issues
- otherwise unknown

foundation_signal:
- positive: new/repaired/engineered, or explicitly solid/dry
- negative: cracks, settling, intrusion, moisture, or known foundation issue
- neutral: foundation mentioned with no condition signal
- unknown: not mentioned

kitchen_quality and bath_quality:
- high-end: premium stone/custom/premium-appliance or luxury-finish language
- updated: renovated or new, without strong premium-material evidence
- standard: described as functional/ordinary without update or defect signal
- dated: original, laminate/Formica, or explicitly needing an update
- poor: damaged or substantially deficient
- unknown: insufficient information

flooring_quality uses the existing common QualityTier schema as follows:
- high-end: predominantly hardwood or another clearly premium treatment
- updated: a positive mix such as hardwood/tile/quality LVP
- standard: ordinary flooring or no meaningful quality signal
- dated: visibly/or explicitly dated flooring
- poor: flooring needs replacement or is materially damaged
- unknown: insufficient information

distress:
- strong: estate sale, bank-owned, must sell, as-is, motivated seller, or
  explicit price-reduction/distress language
- moderate: relocation, pressured downsizing, divorce, or priced-to-sell
- none: no seller-motivation/distress language appears
- unknown only when remarks are missing or unusable

as_is_sale and estate_or_trust_sale:
- true only when expressly supported
- false when that signal is absent from usable remarks

known_defects_advertised:
- true when the public remarks advertise a defect, problem, or material issue
- false when no defect is advertised in usable remarks
- do not use facts found only in the seller disclosure

investor_language:
- true for investment opportunity, rental income, development potential, or
  land-value emphasis
- false when the signal is absent from usable remarks

bucolic_character:
- high: strong emphasis on peace, quiet, rustic setting, wildlife, streams,
  or walkable woods
- moderate: trees, quiet, or rural character mentioned in passing
- low: urban/suburban/in-town language or no rural/bucolic signal
- unknown only when remarks are missing or unusable

historical_character:
- true only for historic/Victorian/period/original architectural character
- false when that signal is absent; "charming" alone is not enough

privacy_high:
- true for private, secluded, set-back, wooded/no-neighbors emphasis
- false when high privacy is not supported

privacy_low:
- true for in-town, close-neighbor, open-lot, or low-privacy language
- false when low privacy is not supported

views_described:
- true only when a scenic, mountain, water, or other view is described
- false when no view is described

lifestyle_tier:
- select the best-supported intended market: luxury, upscale, family,
  starter, retirement, camp-seasonal, investment, or unknown
- use unknown when the intended profile is not reasonably inferable

Call the supplied tool exactly once and return no prose outside the tool call.
""".strip()


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


def _first_public_remarks_evidence(
    record: PropertyRecord,
) -> dict:
    """Return document metadata for the remarks-classification evidence."""

    fact = record.source_text.public_remarks

    if fact.evidence:
        source = fact.evidence[0]

        return {
            "document_id": source.document_id,
            "document_name": source.document_name,
            "page": source.page,
        }

    return {
        "document_id": None,
        "document_name": None,
        "page": None,
    }


def _stamp_remarks_candidate(
    candidate: dict,
    record: PropertyRecord,
    model: str,
) -> dict:
    """
    Add reproducibility metadata and repair only missing evidence metadata.

    Claude still chooses every semantic value. This function does not alter a
    classification. It merely ensures that known classifier outputs carry the
    source and version information needed downstream.
    """

    stamped = deepcopy(candidate)
    stamped["schema_version"] = "remarks-v1"
    stamped["prompt_version"] = REMARKS_PROMPT_VERSION
    stamped["classifier_provider"] = "anthropic"
    stamped["classifier_model"] = model

    document_metadata = _first_public_remarks_evidence(
        record
    )
    remarks = record.source_text.public_remarks.value or ""

    for field_name, field_value in list(stamped.items()):
        if field_name in {
            "schema_version",
            "prompt_version",
            "classifier_provider",
            "classifier_model",
        }:
            continue

        if not isinstance(field_value, dict):
            continue

        if field_value.get("status") != "known":
            continue

        evidence = field_value.get("evidence")

        if not isinstance(evidence, list):
            evidence = []

        if not evidence:
            evidence = [
                {
                    "source_type": SourceType.MLS_REMARKS.value,
                    "document_id": document_metadata["document_id"],
                    "document_name": document_metadata["document_name"],
                    "page": document_metadata["page"],
                    "excerpt": remarks[:240] or None,
                    "extraction_method": ExtractionMethod.LLM.value,
                    "confidence": None,
                    "extractor_name": "remarks_classifier",
                    "extractor_version": REMARKS_PROMPT_VERSION,
                }
            ]

        for item in evidence:
            if not isinstance(item, dict):
                continue

            item["source_type"] = SourceType.MLS_REMARKS.value
            item["extraction_method"] = ExtractionMethod.LLM.value
            item["extractor_name"] = "remarks_classifier"
            item["extractor_version"] = REMARKS_PROMPT_VERSION

            for metadata_name, metadata_value in document_metadata.items():
                if item.get(metadata_name) is None:
                    item[metadata_name] = metadata_value

        field_value["evidence"] = evidence

    return stamped


def _llm_remarks_candidate(
    record: PropertyRecord,
    model: str,
) -> dict | None:
    """
    Classify public remarks with a versioned, training-compatible prompt.

    Returns None when public remarks were not successfully extracted. API and
    schema failures are handled by the caller so a useful property record can
    still be returned.
    """

    remarks_fact = record.source_text.public_remarks

    if (
        remarks_fact.status.value != "known"
        or not remarks_fact.value
        or len(remarks_fact.value.strip()) < 10
    ):
        return None

    remarks = remarks_fact.value.strip()

    if len(remarks) > REMARKS_MAX_CHARS:
        remarks = remarks[:REMARKS_MAX_CHARS]

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    response = client.messages.create(
        model=model,
        max_tokens=5000,
        temperature=0,
        system=REMARKS_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify these public MLS remarks:\n\n"
                    f'"""{remarks}"""'
                ),
            }
        ],
        tools=[
            {
                "name": "record_remarks_classification",
                "description": (
                    "Record training-compatible semantic "
                    "classifications from public MLS remarks."
                ),
                "input_schema": (
                    RemarksClassification.model_json_schema()
                ),
            }
        ],
        tool_choice={
            "type": "tool",
            "name": "record_remarks_classification",
        },
    )

    calls = [
        block
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
        and getattr(block, "name", None)
        == "record_remarks_classification"
    ]

    if len(calls) != 1:
        raise RuntimeError(
            "Expected exactly one remarks-classification "
            f"tool call; received {len(calls)}"
        )

    candidate = calls[0].input

    if not isinstance(candidate, dict):
        raise RuntimeError(
            "Claude remarks-classification output was not "
            "a JSON object"
        )

    return _stamp_remarks_candidate(
        candidate,
        record,
        model,
    )


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

    # Remarks classification is deliberately separate from factual document
    # extraction. A classifier failure must not discard the valid factual
    # record produced above.
    try:
        remarks_candidate = _llm_remarks_candidate(
            record,
            model=model,
        )

        if remarks_candidate is None:
            record.extraction_warnings.append(
                "Remarks classification was not performed because usable "
                "public MLS remarks were not extracted."
            )
            return record

        combined = record.model_dump(
            mode="json",
            exclude_none=False,
        )
        combined["remarks_classification"] = (
            remarks_candidate
        )

        # Reuse the same field-level recovery behavior. One malformed optional
        # classification becomes unknown instead of failing the record.
        return _validate_best_effort(combined)

    except Exception as error:
        record.extraction_warnings.append(
            "Remarks classification failed; factual property extraction "
            "was retained. "
            f"Failure type: {type(error).__name__}."
        )
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
