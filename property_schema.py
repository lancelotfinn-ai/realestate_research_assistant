"""
Canonical property-data contract.

This module contains observable property facts extracted from MLS records,
property disclosures, user statements, and trusted geographic sources.

It deliberately does not contain regression-specific transformations such as:

    log_lot_acres
    feat_basement_quality
    rem_distress_strong
    tract population density
    distance to coast

Those are derived downstream by deterministic feature-engineering code.

Schema version: property-v1
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# COMMON TYPES
# ---------------------------------------------------------------------------

T = TypeVar("T")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class KnowledgeStatus(str, Enum):
    """
    Status of our knowledge, not the value itself.
    """

    KNOWN = "known"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    CONFLICTED = "conflicted"


class SourceType(str, Enum):
    MLS_STRUCTURED = "mls_structured"
    MLS_REMARKS = "mls_remarks"
    PROPERTY_DISCLOSURE = "property_disclosure"
    USER = "user"
    PUBLIC_RECORD = "public_record"
    GEOCODER = "geocoder"
    MAP_DATA = "map_data"
    CALCULATED = "calculated"
    OTHER = "other"


class ExtractionMethod(str, Enum):
    DIRECT_STRUCTURED = "direct_structured"
    PDF_TEXT = "pdf_text"
    PDF_TABLE = "pdf_table"
    OCR = "ocr"
    REGEX = "regex"
    LLM = "llm"
    USER_CONFIRMED = "user_confirmed"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"


class Evidence(StrictModel):
    source_type: SourceType
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page: Optional[int] = Field(None, ge=1)

    field_label: Optional[str] = None
    raw_value: Optional[str] = None
    excerpt: Optional[str] = None

    extraction_method: ExtractionMethod
    confidence: Optional[float] = Field(None, ge=0, le=1)

    # Useful for reproducing LLM classifications.
    extractor_name: Optional[str] = None
    extractor_version: Optional[str] = None


class Fact(StrictModel, Generic[T]):
    """
    A scalar fact with provenance.

    Examples:
        Fact[float](status="known", value=1848, ...)
        Fact[bool](status="known", value=False, ...)
        Fact[bool](status="unknown", value=None, ...)
    """

    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    value: Optional[T] = None
    evidence: list[Evidence] = Field(default_factory=list)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_status_and_value(self):
        if self.status == KnowledgeStatus.KNOWN and self.value is None:
            raise ValueError("A known fact must have a value")

        if (
            self.status in {
                KnowledgeStatus.UNKNOWN,
                KnowledgeStatus.NOT_APPLICABLE,
                KnowledgeStatus.CONFLICTED,
            }
            and self.value is not None
        ):
            raise ValueError(
                f"A {self.status.value} fact should not have a resolved value"
            )

        return self


class ObservedSet(StrictModel, Generic[T]):
    """
    A multi-select field such as heating systems or architectural styles.

    complete=True means the relevant MLS section was observed in full, so an
    unlisted option can safely be encoded as absent.

    complete=False means unlisted options remain unknown.
    """

    values: list[T] = Field(default_factory=list)
    complete: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# ENUMERATED PROPERTY VALUES
# ---------------------------------------------------------------------------

class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"
    CONDOMINIUM = "condominium"
    MOBILE_MANUFACTURED = "mobile_manufactured"
    CAMP_SEASONAL = "camp_seasonal"
    MULTI_FAMILY = "multi_family"
    OTHER = "other"


class HeatingSystem(str, Enum):
    HEAT_PUMP = "heat_pump"
    BASEBOARD = "baseboard"
    HOT_WATER = "hot_water"
    FORCED_AIR = "forced_air"
    WOOD_STOVE = "wood_stove"
    PELLET_STOVE = "pellet_stove"
    RADIANT = "radiant"
    OTHER = "other"
    NONE = "none"


class HeatingFuel(str, Enum):
    OIL = "oil"
    PROPANE = "propane"
    ELECTRIC = "electric"
    WOOD = "wood"
    NATURAL_GAS = "natural_gas"
    PELLETS = "pellets"
    OTHER = "other"


class CoolingSystem(str, Enum):
    HEAT_PUMP = "heat_pump"
    CENTRAL_AIR = "central_air"
    WINDOW_UNITS = "window_units"
    NONE = "none"
    OTHER = "other"


class BasementType(str, Enum):
    FULL = "full"
    FINISHED = "finished"
    UNFINISHED = "unfinished"
    WALKOUT = "walkout"
    DAYLIGHT = "daylight"
    CRAWL_SPACE = "crawl_space"
    SLAB = "slab"
    NONE = "none"
    DIRT_FLOOR = "dirt_floor"
    SUMP_PUMP = "sump_pump"
    OTHER = "other"

class FoundationType(str, Enum):
    POURED_CONCRETE = "poured_concrete"
    CONCRETE_BLOCK = "concrete_block"
    STONE = "stone"
    FIELDSTONE = "fieldstone"
    PIER = "pier"
    SLAB = "slab"
    OTHER = "other"


class RoofMaterial(str, Enum):
    ASPHALT_SHINGLE = "asphalt_shingle"
    METAL = "metal"
    FLAT = "flat"
    OTHER = "other"


class CountertopMaterial(str, Enum):
    GRANITE = "granite"
    QUARTZ = "quartz"
    LAMINATE = "laminate"
    FORMICA = "formica"
    OTHER = "other"


class FlooringMaterial(str, Enum):
    HARDWOOD = "hardwood"
    ENGINEERED_HARDWOOD = "engineered_hardwood"
    CARPET = "carpet"
    TILE = "tile"
    VINYL = "vinyl"
    LUXURY_VINYL = "luxury_vinyl"
    LAMINATE = "laminate"
    LINOLEUM = "linoleum"
    OTHER = "other"


class ArchitecturalStyle(str, Enum):
    RANCH = "ranch"
    CAPE_COD = "cape_cod"
    COLONIAL = "colonial"
    CONTEMPORARY = "contemporary"
    NEW_ENGLANDER = "new_englander"
    COTTAGE = "cottage"
    FARMHOUSE = "farmhouse"
    CAMP = "camp"
    RAISED_RANCH = "raised_ranch"
    OTHER = "other"


class ExteriorMaterial(str, Enum):
    VINYL = "vinyl"
    WOOD = "wood"
    CLAPBOARD = "clapboard"
    SHINGLE = "shingle"
    BRICK = "brick"
    LOG = "log"
    ASBESTOS = "asbestos"
    FIBER_CEMENT = "fiber_cement"
    OTHER = "other"


class WaterAccessType(str, Enum):
    DEEDED = "deeded"
    RIGHT_OF_WAY = "right_of_way"
    NEARBY = "nearby"
    OCEANFRONT = "oceanfront"
    DOCK = "dock"
    OTHER = "other"


class RoadType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    DIRT = "dirt"
    GRAVEL = "gravel"
    SEASONAL = "seasonal"
    PAVED = "paved"
    OTHER = "other"


class LocationTag(str, Enum):
    INTOWN = "intown"
    RURAL = "rural"
    SKI_RESORT = "ski_resort"
    NEAR_PUBLIC_BEACH = "near_public_beach"
    OTHER = "other"


class SiteTag(str, Enum):
    LEVEL = "level"
    WOODED = "wooded"
    CUL_DE_SAC = "cul_de_sac"
    OTHER = "other"


# ---------------------------------------------------------------------------
# IDENTIFICATION AND SOURCE TEXT
# ---------------------------------------------------------------------------

class ListingIdentity(StrictModel):
    mls_number: Fact[str] = Field(default_factory=Fact[str])
    listing_status: Fact[str] = Field(default_factory=Fact[str])
    list_date: Fact[date] = Field(default_factory=Fact[date])
    list_price: Fact[float] = Field(default_factory=Fact[float])
    property_subtype_raw: Fact[str] = Field(default_factory=Fact[str])


class Address(StrictModel):
    full_address: Fact[str] = Field(default_factory=Fact[str])
    street_number: Fact[str] = Field(default_factory=Fact[str])
    street_name: Fact[str] = Field(default_factory=Fact[str])
    unit: Fact[str] = Field(default_factory=Fact[str])
    city: Fact[str] = Field(default_factory=Fact[str])
    state: Fact[str] = Field(default_factory=Fact[str])
    postal_code: Fact[str] = Field(default_factory=Fact[str])
    county: Fact[str] = Field(default_factory=Fact[str])


class Coordinates(StrictModel):
    latitude: Fact[float] = Field(default_factory=Fact[float])
    longitude: Fact[float] = Field(default_factory=Fact[float])

    @model_validator(mode="after")
    def coordinates_must_be_paired(self):
        lat_known = self.latitude.status == KnowledgeStatus.KNOWN
        lon_known = self.longitude.status == KnowledgeStatus.KNOWN

        if lat_known != lon_known:
            raise ValueError(
                "Latitude and longitude must either both be known or neither known"
            )

        if lat_known:
            if not -90 <= self.latitude.value <= 90:
                raise ValueError("Latitude is outside its valid range")
            if not -180 <= self.longitude.value <= 180:
                raise ValueError("Longitude is outside its valid range")

        return self


class SourceText(StrictModel):
    """
    Retain original source material. Never throw away the text after deriving
    model features from it.
    """

    public_remarks: Fact[str] = Field(default_factory=Fact[str])
    private_remarks: Fact[str] = Field(default_factory=Fact[str])
    mls_features_raw: Fact[str] = Field(default_factory=Fact[str])
    disclosure_narratives: list[Fact[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CORE STRUCTURAL FACTS
# ---------------------------------------------------------------------------

class Structure(StrictModel):
    property_type: Fact[PropertyType] = Field(
        default_factory=Fact[PropertyType]
    )

    finished_square_feet: Fact[float] = Field(default_factory=Fact[float])
    lot_acres: Fact[float] = Field(default_factory=Fact[float])
    bedrooms: Fact[int] = Field(default_factory=Fact[int])
    total_bathrooms: Fact[float] = Field(default_factory=Fact[float])
    full_bathrooms: Fact[int] = Field(default_factory=Fact[int])
    half_bathrooms: Fact[int] = Field(default_factory=Fact[int])
    year_built: Fact[int] = Field(default_factory=Fact[int])

    stories: Fact[float] = Field(default_factory=Fact[float])
    units: Fact[int] = Field(default_factory=Fact[int])
    seasonal_use: Fact[bool] = Field(default_factory=Fact[bool])
    new_construction: Fact[bool] = Field(default_factory=Fact[bool])


# ---------------------------------------------------------------------------
# PHYSICAL FEATURES USED BY THE CURRENT MODEL
# ---------------------------------------------------------------------------

class WaterFeatures(StrictModel):
    direct_frontage: Fact[bool] = Field(default_factory=Fact[bool])
    frontage_feet: Fact[float] = Field(default_factory=Fact[float])
    water_view: Fact[bool] = Field(default_factory=Fact[bool])
    seasonal_water_view: Fact[bool] = Field(default_factory=Fact[bool])
    access_types: ObservedSet[WaterAccessType] = Field(
        default_factory=ObservedSet[WaterAccessType]
    )
    water_body_name: Fact[str] = Field(default_factory=Fact[str])
    water_body_type: Fact[str] = Field(default_factory=Fact[str])


class HvacFeatures(StrictModel):
    heating_systems: ObservedSet[HeatingSystem] = Field(
        default_factory=ObservedSet[HeatingSystem]
    )
    heating_fuels: ObservedSet[HeatingFuel] = Field(
        default_factory=ObservedSet[HeatingFuel]
    )
    cooling_systems: ObservedSet[CoolingSystem] = Field(
        default_factory=ObservedSet[CoolingSystem]
    )

    heating_system_year: Fact[int] = Field(default_factory=Fact[int])
    heating_system_condition: Fact[str] = Field(default_factory=Fact[str])


class BasementFoundationFeatures(StrictModel):
    basement_features: ObservedSet[BasementType] = Field(
        default_factory=ObservedSet[BasementType]
    )
    foundation_types: ObservedSet[FoundationType] = Field(
        default_factory=ObservedSet[FoundationType]
    )


class InteriorFeatures(StrictModel):
    countertop_materials: ObservedSet[CountertopMaterial] = Field(
        default_factory=ObservedSet[CountertopMaterial]
    )
    flooring_materials: ObservedSet[FlooringMaterial] = Field(
        default_factory=ObservedSet[FlooringMaterial]
    )

    kitchen_island: Fact[bool] = Field(default_factory=Fact[bool])
    eat_in_kitchen: Fact[bool] = Field(default_factory=Fact[bool])
    primary_bedroom_with_bath: Fact[bool] = Field(
        default_factory=Fact[bool]
    )
    first_floor_laundry: Fact[bool] = Field(default_factory=Fact[bool])
    in_law_apartment: Fact[bool] = Field(default_factory=Fact[bool])
    walk_in_closets: Fact[bool] = Field(default_factory=Fact[bool])


class ExteriorFeatures(StrictModel):
    architectural_styles: ObservedSet[ArchitecturalStyle] = Field(
        default_factory=ObservedSet[ArchitecturalStyle]
    )
    exterior_materials: ObservedSet[ExteriorMaterial] = Field(
        default_factory=ObservedSet[ExteriorMaterial]
    )
    roof_materials: ObservedSet[RoofMaterial] = Field(
        default_factory=ObservedSet[RoofMaterial]
    )

    deck: Fact[bool] = Field(default_factory=Fact[bool])
    porch: Fact[bool] = Field(default_factory=Fact[bool])
    screened_porch: Fact[bool] = Field(default_factory=Fact[bool])
    in_ground_pool: Fact[bool] = Field(default_factory=Fact[bool])
    barn: Fact[bool] = Field(default_factory=Fact[bool])
    outbuilding: Fact[bool] = Field(default_factory=Fact[bool])


class GarageFeatures(StrictModel):
    garage_spaces: Fact[float] = Field(default_factory=Fact[float])
    attached: Fact[bool] = Field(default_factory=Fact[bool])
    detached: Fact[bool] = Field(default_factory=Fact[bool])
    direct_entry: Fact[bool] = Field(default_factory=Fact[bool])
    heated: Fact[bool] = Field(default_factory=Fact[bool])
    storage_above: Fact[bool] = Field(default_factory=Fact[bool])
    explicitly_no_vehicle_storage: Fact[bool] = Field(
        default_factory=Fact[bool]
    )


class UtilitiesEquipment(StrictModel):
    public_water: Fact[bool] = Field(default_factory=Fact[bool])
    public_sewer: Fact[bool] = Field(default_factory=Fact[bool])
    natural_gas_available: Fact[bool] = Field(default_factory=Fact[bool])

    generator: Fact[bool] = Field(default_factory=Fact[bool])
    generator_hookup: Fact[bool] = Field(default_factory=Fact[bool])
    radon_air_mitigation: Fact[bool] = Field(default_factory=Fact[bool])
    seller_owned_solar: Fact[bool] = Field(default_factory=Fact[bool])
    double_pane_windows: Fact[bool] = Field(default_factory=Fact[bool])
    internet_available: Fact[bool] = Field(default_factory=Fact[bool])


class AccessLocationFeatures(StrictModel):
    road_types: ObservedSet[RoadType] = Field(
        default_factory=ObservedSet[RoadType]
    )
    paved_driveway: Fact[bool] = Field(default_factory=Fact[bool])
    no_driveway: Fact[bool] = Field(default_factory=Fact[bool])

    location_tags: ObservedSet[LocationTag] = Field(
        default_factory=ObservedSet[LocationTag]
    )
    site_tags: ObservedSet[SiteTag] = Field(
        default_factory=ObservedSet[SiteTag]
    )

    scenic_view: Fact[bool] = Field(default_factory=Fact[bool])
    mountain_view: Fact[bool] = Field(default_factory=Fact[bool])


# ---------------------------------------------------------------------------
# DISCLOSURE FACTS
# ---------------------------------------------------------------------------

class ConditionSignal(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class DisclosureFacts(StrictModel):
    """
    Factual disclosure information is kept separate from remarks-derived model
    classifications. A disclosure defect is not automatically equivalent to
    rem_known_defects, because the latter was trained from public remarks.
    """

    roof_installation_year: Fact[int] = Field(default_factory=Fact[int])
    windows_installation_year: Fact[int] = Field(default_factory=Fact[int])
    heating_installation_year: Fact[int] = Field(default_factory=Fact[int])
    septic_installation_year: Fact[int] = Field(default_factory=Fact[int])
    septic_last_pumped_date: Fact[date] = Field(default_factory=Fact[date])

    active_roof_leak: Fact[bool] = Field(default_factory=Fact[bool])
    past_roof_leak: Fact[bool] = Field(default_factory=Fact[bool])
    basement_water_intrusion: Fact[bool] = Field(default_factory=Fact[bool])
    foundation_problem: Fact[bool] = Field(default_factory=Fact[bool])
    mold_reported: Fact[bool] = Field(default_factory=Fact[bool])
    radon_issue_reported: Fact[bool] = Field(default_factory=Fact[bool])
    septic_problem: Fact[bool] = Field(default_factory=Fact[bool])
    hazardous_material_reported: Fact[bool] = Field(
        default_factory=Fact[bool]
    )

    material_defects: list[Fact[str]] = Field(default_factory=list)
    improvements: list[Fact[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TRAINING-COMPATIBLE REMARKS CLASSIFICATION
# ---------------------------------------------------------------------------

class RemarksCondition(str, Enum):
    MOVE_IN_READY = "move-in-ready"
    UPDATED = "updated"
    DATED = "dated"
    NEEDS_WORK = "needs-work"
    FIXER = "fixer"
    UNKNOWN = "unknown"


class QualityTier(str, Enum):
    HIGH_END = "high-end"
    UPDATED = "updated"
    STANDARD = "standard"
    DATED = "dated"
    POOR = "poor"
    UNKNOWN = "unknown"


class BucolicTier(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class DistressTier(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    NONE = "none"
    UNKNOWN = "unknown"


class LifestyleTier(str, Enum):
    LUXURY = "luxury"
    UPSCALE = "upscale"
    FAMILY = "family"
    STARTER = "starter"
    RETIREMENT = "retirement"
    CAMP_SEASONAL = "camp-seasonal"
    INVESTMENT = "investment"
    UNKNOWN = "unknown"


class RemarksClassification(StrictModel):
    """
    Semantic results from the versioned remarks classifier.

    These are still not the final rem_* regression columns. R converts this
    object using the same encoding rules used during training.
    """

    schema_version: str = "remarks-v1"
    prompt_version: Optional[str] = None
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None

    condition: Fact[RemarksCondition] = Field(
        default_factory=Fact[RemarksCondition]
    )

    new_roof: Fact[bool] = Field(default_factory=Fact[bool])
    new_heating: Fact[bool] = Field(default_factory=Fact[bool])
    new_windows: Fact[bool] = Field(default_factory=Fact[bool])
    new_basement_work: Fact[bool] = Field(default_factory=Fact[bool])
    systems_updated: Fact[bool] = Field(default_factory=Fact[bool])
    water_issues: Fact[bool] = Field(default_factory=Fact[bool])

    foundation_signal: Fact[ConditionSignal] = Field(
        default_factory=Fact[ConditionSignal]
    )

    kitchen_quality: Fact[QualityTier] = Field(
        default_factory=Fact[QualityTier]
    )
    bath_quality: Fact[QualityTier] = Field(
        default_factory=Fact[QualityTier]
    )
    flooring_quality: Fact[QualityTier] = Field(
        default_factory=Fact[QualityTier]
    )

    distress: Fact[DistressTier] = Field(
        default_factory=Fact[DistressTier]
    )
    as_is_sale: Fact[bool] = Field(default_factory=Fact[bool])
    estate_or_trust_sale: Fact[bool] = Field(default_factory=Fact[bool])
    known_defects_advertised: Fact[bool] = Field(default_factory=Fact[bool])
    investor_language: Fact[bool] = Field(default_factory=Fact[bool])

    bucolic_character: Fact[BucolicTier] = Field(
        default_factory=Fact[BucolicTier]
    )
    historical_character: Fact[bool] = Field(default_factory=Fact[bool])
    privacy_high: Fact[bool] = Field(default_factory=Fact[bool])
    privacy_low: Fact[bool] = Field(default_factory=Fact[bool])
    views_described: Fact[bool] = Field(default_factory=Fact[bool])

    lifestyle_tier: Fact[LifestyleTier] = Field(
        default_factory=Fact[LifestyleTier]
    )


# ---------------------------------------------------------------------------
# CONFLICTS AND COMPLETE CANONICAL RECORD
# ---------------------------------------------------------------------------

class FieldConflict(StrictModel):
    field_path: str
    candidates: list[Fact[Any]]
    resolution: Optional[str] = None
    requires_user_review: bool = True


class PropertyRecord(StrictModel):
    schema_version: str = "property-v1"

    listing: ListingIdentity = Field(default_factory=ListingIdentity)
    address: Address = Field(default_factory=Address)
    coordinates: Coordinates = Field(default_factory=Coordinates)
    source_text: SourceText = Field(default_factory=SourceText)

    structure: Structure = Field(default_factory=Structure)
    water: WaterFeatures = Field(default_factory=WaterFeatures)
    hvac: HvacFeatures = Field(default_factory=HvacFeatures)
    basement_foundation: BasementFoundationFeatures = Field(
        default_factory=BasementFoundationFeatures
    )
    interior: InteriorFeatures = Field(default_factory=InteriorFeatures)
    exterior: ExteriorFeatures = Field(default_factory=ExteriorFeatures)
    garage: GarageFeatures = Field(default_factory=GarageFeatures)
    utilities_equipment: UtilitiesEquipment = Field(
        default_factory=UtilitiesEquipment
    )
    access_location: AccessLocationFeatures = Field(
        default_factory=AccessLocationFeatures
    )

    disclosures: DisclosureFacts = Field(default_factory=DisclosureFacts)
    remarks_classification: Optional[RemarksClassification] = None

    conflicts: list[FieldConflict] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
