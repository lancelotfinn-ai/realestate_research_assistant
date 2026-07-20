from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from document_ingest import DocumentArtifact


MONEY_RE = re.compile(r"[^0-9.]", re.ASCII)


def _number(value: str) -> float:
    return float(MONEY_RE.sub("", value))


def _integer(value: str) -> int:
    return int(round(_number(value)))


def _iso_date(value: str) -> str:
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date: {value}")


def _evidence(doc: DocumentArtifact, page: int, label: str, raw: str) -> dict:
    return {
        "source_type": "mls_structured",
        "document_id": doc.document_id,
        "document_name": doc.name,
        "page": page,
        "field_label": label,
        "raw_value": raw.strip(),
        "extraction_method": "regex",
        "confidence": 0.99,
        "extractor_name": "deterministic_mls",
        "extractor_version": "1",
    }


def _known(value: Any, evidence: dict) -> dict:
    return {"status": "known", "value": value, "evidence": [evidence]}


# These aliases should be expanded from actual MLS exports as new layouts appear.
SCALAR_PATTERNS: tuple[tuple[str, str, Callable[[str], Any]], ...] = (
    ("listing.mls_number", r"(?:MLS|List)\s*(?:#|Number)\s*[:#]?\s*([A-Za-z0-9-]+)", str),
    ("listing.list_price", r"List\s*Price\s*[:$]?\s*([$0-9,]+(?:\.\d+)?)", _number),
    ("listing.list_date", r"List\s*Date\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})", _iso_date),
    ("structure.finished_square_feet", r"(?:Sq\.?\s*Ft\.?\s*Finished\s*Total|Finished\s*Sq\.?\s*Ft\.?)\s*:?\s*([0-9,]+)", _number),
    ("structure.lot_acres", r"Lot\s*(?:Size\s*)?(?:Acres|Acreage)\s*:?\s*([0-9,.]+)", _number),
    ("structure.bedrooms", r"(?:Bedrooms|Beds)\s*:?\s*(\d+)", _integer),
    ("structure.total_bathrooms", r"(?:Total\s*Baths|Bathrooms|Baths)\s*:?\s*([0-9.]+)", _number),
    ("structure.year_built", r"Year\s*Built\s*:?\s*(\d{4})", _integer),
    ("coordinates.latitude", r"(?:Geo\.?\s*Lat|Latitude)\s*:?\s*(-?\d{1,2}\.\d+)", _number),
    ("coordinates.longitude", r"(?:Geo\.?\s*Lon|Longitude)\s*:?\s*(-?\d{1,3}\.\d+)", _number),
)


def _set_path(target: dict, dotted_path: str, value: dict) -> None:
    parts = dotted_path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def extract_mls_scalars(document: DocumentArtifact) -> dict:
    """
    Extract only high-confidence labeled scalar fields.

    If a PDF has broken font encoding, these fields remain absent and the vision
    LLM pass may recover them. This function deliberately does not guess.
    """
    result: dict = {}

    for page in document.pages:
        for path, pattern, converter in SCALAR_PATTERNS:
            if _get_path(result, path) is not None:
                continue
            match = re.search(pattern, page.text, flags=re.I)
            if not match:
                continue
            raw = match.group(1)
            try:
                converted = converter(raw)
            except (TypeError, ValueError):
                continue
            label = pattern.split("\\s")[0].strip("(?:")
            _set_path(
                result,
                path,
                _known(converted, _evidence(document, page.page_number, label, raw)),
            )

    # Do not emit a half-coordinate pair. It is safer to leave geography
    # unknown and allow the LLM/geocoder stage to supply both values together.
    lat = _get_path(result, "coordinates.latitude")
    lon = _get_path(result, "coordinates.longitude")
    if (lat is None) != (lon is None):
        result.pop("coordinates", None)

    return result


def _get_path(target: dict, dotted_path: str):
    cursor = target
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor
