import os
import json
import secrets
import subprocess
import tempfile
import time
import select
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

import requests
from bs4 import BeautifulSoup
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from extract_property import build_property_record
from property_schema import PropertyRecord


# ============================================================
# PERSISTENT R VALUATION WORKER
# ============================================================

class RWorker:
    """
    Keeps one R process alive so the model and geographic
    artifacts do not need to be reloaded for every request.
    """

    def __init__(
        self,
        cmd=None,
        ready_timeout=90,
        call_timeout=45,
    ):
        self.cmd = cmd or ["Rscript", "r_worker.R"]
        self.ready_timeout = ready_timeout
        self.call_timeout = call_timeout

        self.proc = None
        self._ready = False
        self._buf = b""
        self._counter = 0
        self.lock = threading.Lock()

    def start(self):
        self.stop()

        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Let R diagnostics appear in Render logs. The worker reserves
            # stdout exclusively for its newline-delimited JSON protocol.
            stderr=None,
            bufsize=0,
        )

        self._ready = False
        self._buf = b""

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        self.proc = None
        self._ready = False
        self._buf = b""

    def _alive(self):
        return (
            self.proc is not None
            and self.proc.poll() is None
        )

    def _read_line(self, deadline):
        while True:
            newline_position = self._buf.find(b"\n")

            if newline_position != -1:
                line = self._buf[:newline_position]
                self._buf = self._buf[
                    newline_position + 1:
                ]

                return line.decode(
                    "utf-8",
                    "replace",
                ).strip()

            remaining = deadline - time.time()

            if remaining <= 0:
                raise TimeoutError(
                    "R worker read timed out"
                )

            readable, _, _ = select.select(
                [self.proc.stdout],
                [],
                [],
                remaining,
            )

            if not readable:
                raise TimeoutError(
                    "R worker read timed out"
                )

            chunk = os.read(
                self.proc.stdout.fileno(),
                65536,
            )

            if chunk == b"":
                raise RuntimeError(
                    "R worker closed stdout "
                    "(process died)"
                )

            self._buf += chunk

    def _ensure_ready(self):
        if self._ready:
            return

        deadline = time.time() + self.ready_timeout

        while True:
            line = self._read_line(deadline)

            if not line:
                continue

            try:
                response = json.loads(line)
            except ValueError:
                continue

            if response.get("ready"):
                self._ready = True
                return

    def value(
        self,
        property_record: dict,
        asof: Optional[str] = None,
    ):
        """
        Value one canonical property-v1 record with the persistent R worker.
        """

        with self.lock:
            if not self._alive():
                self.start()

            try:
                self._ensure_ready()

                self._counter += 1
                request_id = self._counter

                request_data = {
                    "id": request_id,
                    "property_record": property_record,
                    "asof": asof,
                }

                request_line = (
                    json.dumps(request_data) + "\n"
                )

                self.proc.stdin.write(
                    request_line.encode("utf-8")
                )
                self.proc.stdin.flush()

                deadline = (
                    time.time() + self.call_timeout
                )

                while True:
                    line = self._read_line(deadline)

                    if not line:
                        continue

                    try:
                        response = json.loads(line)
                    except ValueError:
                        continue

                    if response.get("id") == request_id:
                        return response

            except Exception:
                self.stop()
                raise


r_worker = RWorker()


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app):
    try:
        r_worker.start()
        print("[rworker] spawned successfully")
    except Exception as error:
        print(
            "[rworker] could not spawn; "
            f"CLI fallback will be used: {error}"
        )

    yield

    r_worker.stop()


app = FastAPI(
    lifespan=lifespan,
    title="Maine Housing Analytics Engine",
    description=(
        "Evidence-backed Maine property extraction and full-model valuation "
        "API for AI tool calling."
    ),
    version="3.0.0",
)


# ============================================================
# DOCUMENT-INGESTION CONFIGURATION
# ============================================================

# This limit applies separately to the MLS PDF and disclosure
# PDF. Files are streamed to temporary storage rather than read
# into memory in one operation.
MAX_PDF_BYTES = 20 * 1024 * 1024


def _verify_ingestion_key(
    supplied_key: Optional[str],
):
    """
    Protect the Claude-backed document endpoint from public use.

    INGESTION_API_KEY is an application-level credential chosen
    by the service owner. It is distinct from ANTHROPIC_API_KEY,
    which must never be sent by a client.
    """

    expected_key = os.getenv("INGESTION_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Document ingestion is not configured: "
                "INGESTION_API_KEY is missing"
            ),
        )

    if (
        not supplied_key
        or not secrets.compare_digest(
            supplied_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid ingestion API key",
            headers={"WWW-Authenticate": "X-Ingestion-Key"},
        )


async def _save_pdf_upload(
    upload: UploadFile,
    destination: Path,
):
    """
    Stream an uploaded PDF to a request-scoped temporary file.

    Content-Type is checked as an early diagnostic, and the PDF
    file signature is checked after writing. The caller owns the
    temporary directory and removes it after extraction.
    """

    allowed_content_types = {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    }

    if upload.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{upload.filename or 'Uploaded file'} "
                "must be a PDF"
            ),
        )

    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_PDF_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{upload.filename or 'Uploaded file'} "
                            "exceeds the 20 MB limit"
                        ),
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{upload.filename or 'Uploaded file'} "
                    "is empty"
                ),
            )

        with destination.open("rb") as saved_file:
            if saved_file.read(5) != b"%PDF-":
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"{upload.filename or 'Uploaded file'} "
                        "does not have a valid PDF signature"
                    ),
                )

    finally:
        await upload.close()


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ModelSummary(BaseModel):
    description: Optional[str] = None
    training_period: Optional[str] = None
    n_observations: Optional[int] = None
    r_squared: Optional[float] = None


class GeographySummary(BaseModel):
    source: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    tract_fips: Optional[str] = None
    tract_matched: Optional[bool] = None
    predictors: dict[str, Optional[float]] = Field(default_factory=dict)


class ValuationDriver(BaseModel):
    variable: str
    label: str
    dollar_effect: float
    pct_effect: float
    comparison: str


class ValuationNarrative(BaseModel):
    summary: str
    principal_drivers: list[str] = Field(default_factory=list)
    geography: str
    disclosure_caveats: list[str] = Field(default_factory=list)
    limitations: str


class InputDiagnostics(BaseModel):
    input_mode: Optional[str] = None
    n_provided: int
    supplied_variables: list[str] = Field(default_factory=list)
    imputed_impact_share: float
    imputed_variables: list[str] = Field(default_factory=list)
    ignored_variables: list[str] = Field(default_factory=list)


class ValuationResponse(BaseModel):
    estimate: float
    range_low: float
    range_high: float
    range_method: str
    as_of: date
    model: ModelSummary
    geography: GeographySummary
    input_diagnostics: InputDiagnostics
    drivers: list[ValuationDriver] = Field(default_factory=list)
    narrative: ValuationNarrative
    suggested_follow_up_questions: list[str] = Field(default_factory=list)


class ListingFetchRequest(BaseModel):
    url: str = Field(
        ...,
        description=(
            "Public real-estate listing URL, such "
            "as an MLS, brokerage, or portal page."
        ),
    )


# ============================================================
# RESPONSE SHAPING
# ============================================================

def _shape_valuation(result):
    """
    Convert the R worker result into the documented public response without
    discarding model provenance, drivers, narrative, or missing-input detail.
    """

    if not result or not result.get("ok"):
        reason = (
            result.get("reason")
            if result
            else None
        )

        status_code = 422 if reason in {
            "could_not_resolve_geography",
            "could_not_geocode",
        } else 500

        raise HTTPException(
            status_code=status_code,
            detail=reason or "The valuation engine did not return a result",
        )

    geographic_predictors = (
        result.get("geographic_features") or {}
    )

    # jsonlite serializes a one-row R data.frame as a one-element array when
    # dataframe="rows". Expose the row itself in the public API.
    if (
        isinstance(geographic_predictors, list)
        and len(geographic_predictors) == 1
        and isinstance(geographic_predictors[0], dict)
    ):
        geographic_predictors = geographic_predictors[0]

    return {
        "estimate": result["estimate"],
        "range_low": result["low"],
        "range_high": result["high"],
        "range_method": result.get("range_method"),
        "as_of": result.get("asof"),
        "model": result.get("model") or {},

        "geography": {
            "source": result.get(
                "geography_source"
            ),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "tract_fips": result.get(
                "tract_fips"
            ),
            "tract_matched": result.get(
                "tract_matched"
            ),
            "predictors": geographic_predictors,
        },

        "input_diagnostics": {
            "n_provided": result.get(
                "n_provided"
            ),
            "input_mode": result.get(
                "input_mode"
            ),
            "supplied_variables": result.get(
                "supplied_variables"
            ) or [],
            "imputed_impact_share": result.get(
                "imputed_impact_share"
            ),
            "imputed_variables": result.get(
                "imputed_variables"
            ) or [],
            "ignored_variables": result.get(
                "ignored_variables"
            ) or [],
        },

        "drivers": result.get("drivers") or [],
        "narrative": result.get("narrative") or {},

        "suggested_follow_up_questions": (
            result.get(
                "suggest_asking_about",
                [],
            )
        ),
    }


# ============================================================
# CLI FALLBACK
# ============================================================

def _valuation_cli(
    property_record: dict,
    asof: Optional[str] = None,
):
    """
    Run one fresh R process if the persistent worker fails.
    """

    try:
        payload = json.dumps({
            "property_record": property_record,
            "asof": asof,
        })

        process = subprocess.run(
            ["Rscript", "valuation.R", payload],
            capture_output=True,
            text=True,
            timeout=45,
        )

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="Fallback valuation process failed",
            )

        result = json.loads(process.stdout)

        return _shape_valuation(result)

    except Exception as error:
        if isinstance(error, HTTPException):
            raise
        raise HTTPException(
            status_code=503,
            detail="The valuation engine is unavailable",
        ) from error


# ============================================================
# SHARED VALUATION LOGIC
# ============================================================

def _run_valuation(
    record: PropertyRecord,
    as_of: Optional[date] = None,
):
    """
    Pass a validated canonical property record to R. All feature engineering
    stays in valuation.R so the training encodings have one implementation.
    """
    property_record = record.model_dump(
        mode="json",
        exclude_none=False,
    )
    asof = as_of.isoformat() if as_of else None

    try:
        result = r_worker.value(
            property_record=property_record,
            asof=asof,
        )

    except Exception as error:
        print(
            "[valuation] background worker failed "
            f"({error}); invoking CLI fallback..."
        )

        return _valuation_cli(
            property_record=property_record,
            asof=asof,
        )

    # Keep model/input errors returned by a healthy worker distinct from
    # transport failures. _shape_valuation converts them to an appropriate
    # HTTP response without rerunning the same invalid request in a new R
    # process.
    return _shape_valuation(result)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "r_worker_alive": r_worker._alive(),
        "document_extraction": {
            "anthropic_key_configured": bool(
                os.getenv("ANTHROPIC_API_KEY")
            ),
            "anthropic_model_configured": bool(
                os.getenv("ANTHROPIC_MODEL")
            ),
            "ingestion_auth_configured": bool(
                os.getenv("INGESTION_API_KEY")
            ),
        },
    }


@app.post(
    "/extract_property_record",
    response_model=PropertyRecord,
)
async def extract_property_record(
    mls: Annotated[
        UploadFile,
        File(
            ...,
            description="MLS listing PDF",
        ),
    ],
    disclosure: Annotated[
        UploadFile,
        File(
            ...,
            description="Seller property-disclosure PDF",
        ),
    ],
    x_ingestion_key: Annotated[
        Optional[str],
        Header(
            description=(
                "Application-level credential for the "
                "document-ingestion endpoint"
            ),
        ),
    ] = None,
):
    """
    Extract an evidence-backed canonical PropertyRecord from an
    MLS listing PDF and seller disclosure PDF.

    The documents are stored only in a request-scoped temporary
    directory. PDF rendering, the Anthropic request, merging, and
    Pydantic validation are delegated to build_property_record().
    """

    _verify_ingestion_key(x_ingestion_key)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Document ingestion is not configured: "
                "ANTHROPIC_API_KEY is missing"
            ),
        )

    anthropic_model = os.getenv("ANTHROPIC_MODEL")

    if not anthropic_model:
        raise HTTPException(
            status_code=503,
            detail=(
                "Document ingestion is not configured: "
                "ANTHROPIC_MODEL is missing"
            ),
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="property-extraction-",
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            mls_path = temporary_path / "mls.pdf"
            disclosure_path = (
                temporary_path / "disclosure.pdf"
            )

            await _save_pdf_upload(
                mls,
                mls_path,
            )
            await _save_pdf_upload(
                disclosure,
                disclosure_path,
            )

            # PDF rendering and the external Claude request are
            # blocking operations. Run them outside FastAPI's
            # asynchronous event loop.
            record = await run_in_threadpool(
                build_property_record,
                str(mls_path),
                str(disclosure_path),
                anthropic_model,
            )

            return record

    except HTTPException:
        raise

    except Exception as error:
        # Keep operational detail in Render logs without returning
        # credentials, document text, or internal paths to clients.
        print(
            "[document-extraction] failed: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Property document extraction failed. "
                "See service logs for the internal error."
            ),
        ) from error


@app.post(
    "/estimate_home_value",
    response_model=ValuationResponse,
)
def estimate_home_value(
    record: PropertyRecord,
    as_of: Optional[date] = None,
):
    """
    Apply the full Maine hedonic model to a canonical PropertyRecord.

    The request body is the property-v1 JSON returned by
    /extract_property_record. Supplied coordinates take priority over address
    geocoding. When as_of is omitted, the current date is used.
    """

    return _run_valuation(record, as_of)


@app.post("/fetch_listing_specs")
def fetch_listing_specs(
    req: ListingFetchRequest,
):
    """
    Retrieve basic metadata and JSON-LD blocks from a public
    real-estate listing webpage.

    This endpoint does not yet normalize the page into a complete
    PropertyRecord.
    """

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(
            req.url,
            headers=headers,
            timeout=20,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    "Listing provider blocked "
                    "automated access"
                ),
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        def meta(key, attr="property"):
            tag = soup.find(
                "meta",
                {attr: key},
            )

            if tag and tag.has_attr("content"):
                return tag["content"]

            return None

        data = {
            "title": (
                soup.title.string
                if soup.title
                else None
            ),
            "og_title": meta("og:title"),
            "og_description": meta(
                "og:description"
            ),
            "description": meta(
                "description",
                attr="name",
            ),
            "extracted_structured_blocks": [],
        }

        for script in soup.find_all(
            "script",
            {"type": "application/ld+json"},
        ):
            text = (script.string or "").strip()

            if text:
                data[
                    "extracted_structured_blocks"
                ].append(text[:4000])

        if not any([
            data["title"],
            data["og_description"],
            data["extracted_structured_blocks"],
        ]):
            return {
                "error": (
                    "webpage loaded but its content "
                    "was unreadable by the scraper"
                )
            }

        return data

    except HTTPException:
        raise

    except Exception as error:
        return {
            "error": (
                "web scraper exception encountered: "
                f"{error}"
            )
        }
