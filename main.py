import os
import json
import subprocess
import time
import select
import threading
from contextlib import asynccontextmanager
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


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
            stderr=subprocess.DEVNULL,
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
        address,
        user_input,
        geo_override=None,
    ):
        """
        Send one valuation request to the persistent R worker.

        geo_override may contain:

        {
            "lat": 44.416855,
            "lon": -68.728069,
            "tract_fips": "23009966400"
        }
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
                    "address": address,
                    "user_input": user_input,
                    "geo_override": geo_override,
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
        "Backend property-valuation API for "
        "AI function calling."
    ),
    version="2.1.0",
)


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ValuationRequest(BaseModel):
    address: Optional[str] = Field(
        None,
        description=(
            "Street address or town in Maine. "
            "Optional when latitude and longitude "
            "are supplied."
        ),
    )

    latitude: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description=(
            "Property latitude, preferably from "
            "the MLS record."
        ),
    )

    longitude: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description=(
            "Property longitude, preferably from "
            "the MLS record."
        ),
    )

    tract_fips: Optional[str] = Field(
        None,
        pattern=r"^\d{11}$",
        description=(
            "Optional 11-digit 2020 Census tract "
            "GEOID. If omitted, the service will "
            "attempt to derive it from coordinates."
        ),
    )

    square_feet: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Total finished living area in "
            "square feet."
        ),
    )

    bedrooms: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of bedrooms.",
    )

    bathrooms: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Total bathrooms; half-baths count "
            "as 0.5."
        ),
    )

    lot_acres: Optional[float] = Field(
        None,
        ge=0,
        description="Lot size in acres.",
    )

    year_built: Optional[int] = Field(
        None,
        ge=1600,
        le=2100,
        description=(
            "Year the home was constructed."
        ),
    )

    is_mobile_home: Optional[bool] = Field(
        None,
        description=(
            "True for a mobile, manufactured, "
            "or double-wide home."
        ),
    )

    is_condo: Optional[bool] = Field(
        None,
        description=(
            "True if the property is a condominium."
        ),
    )

    water_view: Optional[bool] = Field(
        None,
        description=(
            "True if the property has seasonal "
            "or year-round water views."
        ),
    )

    water_frontage: Optional[bool] = Field(
        None,
        description=(
            "True if the property has direct "
            "water or tidal frontage."
        ),
    )

    @model_validator(mode="after")
    def validate_location(self):
        has_address = bool(
            self.address and self.address.strip()
        )

        has_latitude = self.latitude is not None
        has_longitude = self.longitude is not None

        if has_latitude != has_longitude:
            raise ValueError(
                "latitude and longitude must be "
                "supplied together"
            )

        if not has_address and not (
            has_latitude and has_longitude
        ):
            raise ValueError(
                "provide either an address or both "
                "latitude and longitude"
            )

        if self.tract_fips is not None and not (
            has_latitude and has_longitude
        ):
            raise ValueError(
                "tract_fips may only be supplied "
                "with latitude and longitude"
            )

        return self


class ListingFetchRequest(BaseModel):
    url: str = Field(
        ...,
        description=(
            "Public real-estate listing URL, such "
            "as an MLS, brokerage, or portal page."
        ),
    )


# ============================================================
# MODEL FIELD MAPPINGS
# ============================================================

FIELD_MAP = {
    "square_feet": "SqFt.Finished.Total",
    "bedrooms": "X..Bedrooms",
    "bathrooms": "Total.Baths",
    "lot_acres": "Lot.Size.Acres....",
    "year_built": "Year.Built",
}

BOOL_MAP = {
    "is_mobile_home": "is_mh",
    "is_condo": "is_condo",
    "water_view": "feat_water_view",
    "water_frontage": "feat_water_frontage",
}


# ============================================================
# RESPONSE SHAPING
# ============================================================

def _shape_valuation(result):
    """
    Convert the internal R response into the public API response.

    Geographic and input diagnostics are retained so callers
    can verify how the estimate was produced.
    """

    if not result or not result.get("ok"):
        reason = (
            result.get("reason")
            if result
            else None
        )

        response = {
            "error": (
                reason
                or "could not estimate valuation data"
            )
        }

        if result:
            response["geography_source"] = result.get(
                "geography_source"
            )

        return response

    return {
        "estimate": result["estimate"],
        "range_low": result["low"],
        "range_high": result["high"],
        "as_of": result.get("asof"),

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
        },

        "input_diagnostics": {
            "n_provided": result.get(
                "n_provided"
            ),
            "imputed_impact_share": result.get(
                "imputed_impact_share"
            ),
        },

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
    address,
    user_input,
    geo_override=None,
):
    """
    Run one fresh R process if the persistent worker fails.
    """

    try:
        payload = json.dumps({
            "address": address,
            "user_input": user_input,
            "geo_override": geo_override,
        })

        process = subprocess.run(
            ["Rscript", "valuation.R", payload],
            capture_output=True,
            text=True,
            timeout=45,
        )

        if process.returncode != 0:
            return {
                "error": (
                    "fallback valuation script failed"
                ),
                "detail": (
                    process.stderr.strip() or None
                ),
            }

        result = json.loads(process.stdout)

        return _shape_valuation(result)

    except Exception as error:
        return {
            "error": (
                "valuation engine completely "
                f"unavailable: {error}"
            )
        }


# ============================================================
# SHARED VALUATION LOGIC
# ============================================================

def _run_valuation(req: ValuationRequest):
    """
    Translate public request fields into model fields and run
    the valuation. Coordinates take priority over address
    geocoding.
    """

    fields = req.model_dump()
    user_input = {}

    for public_name, model_name in FIELD_MAP.items():
        value = fields.get(public_name)

        if value is not None:
            user_input[model_name] = value

    for public_name, model_name in BOOL_MAP.items():
        value = fields.get(public_name)

        if value is not None:
            user_input[model_name] = (
                1 if value else 0
            )

    geo_override = None

    if (
        req.latitude is not None
        and req.longitude is not None
    ):
        geo_override = {
            "lat": req.latitude,
            "lon": req.longitude,
            "tract_fips": req.tract_fips,
        }

    try:
        result = r_worker.value(
            address=req.address,
            user_input=user_input,
            geo_override=geo_override,
        )

        return _shape_valuation(result)

    except Exception as error:
        print(
            "[valuation] background worker failed "
            f"({error}); invoking CLI fallback..."
        )

        return _valuation_cli(
            address=req.address,
            user_input=user_input,
            geo_override=geo_override,
        )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "r_worker_alive": r_worker._alive(),
    }


@app.post("/estimate_home_value")
def estimate_home_value(
    req: ValuationRequest,
):
    """
    Calculate a model-based market-value estimate.

    Supplied coordinates take priority over Census
    street-address geocoding.
    """

    return _run_valuation(req)


@app.post("/fetch_listing_specs")
def fetch_listing_specs(
    req: ListingFetchRequest,
):
    """
    Retrieve basic metadata and JSON-LD blocks from a public
    real-estate listing webpage.

    This endpoint does not yet normalize the page into a complete
    ValuationRequest.
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
