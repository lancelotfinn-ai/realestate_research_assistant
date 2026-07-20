# ============================================================
# geo_features.R
#
# Geography utilities for the live valuation engine.
#
# Produces the eight geographic predictors used by the model:
#
#   Geo.Lat
#   Geo.Lon
#   pop_density
#   poverty_rate
#   bachelors_plus_rate
#   owner_occ_rate
#   dist_grocery_miles
#   dist_coast_miles
#
# Geography may originate from either:
#
#   1. Latitude and longitude supplied by an MLS record, or
#   2. A street address resolved by the Census Geocoder.
#
# When coordinates are supplied, the Census coordinate endpoint
# is used only to identify the 2020 census tract. It does not
# attempt to geocode the street address.
#
# If the census tract cannot be identified or does not exist in
# the stored ACS lookup, statewide median ACS values are used.
# The supplied coordinates are still used for the spatial model
# terms and distance calculations.
# ============================================================

suppressMessages(
  library(sf)
)


# ============================================================
# COORDINATE REFERENCE SYSTEMS
# ============================================================

# WGS84 latitude and longitude, used by Census and MLS systems.

CRS_GEO <- 4326

# UTM Zone 19N, suitable for distance calculations in Maine.
# The grocery-store artifact is expected to use this CRS.

CRS_PROJ <- 32619


# ============================================================
# BASIC VALIDATION HELPERS
# ============================================================

is_valid_coordinate_number <- function(x) {
  if (is.null(x) || length(x) != 1 || is.na(x)) {
    return(FALSE)
  }

  value <- suppressWarnings(as.numeric(x))

  !is.na(value) && is.finite(value)
}


is_valid_latitude <- function(x) {
  if (!is_valid_coordinate_number(x)) {
    return(FALSE)
  }

  value <- as.numeric(x)

  value >= -90 && value <= 90
}


is_valid_longitude <- function(x) {
  if (!is_valid_coordinate_number(x)) {
    return(FALSE)
  }

  value <- as.numeric(x)

  value >= -180 && value <= 180
}


is_valid_tract_fips <- function(x) {
  if (is.null(x) || length(x) != 1 || is.na(x)) {
    return(FALSE)
  }

  value <- trimws(as.character(x))

  grepl("^[0-9]{11}$", value)
}


# ============================================================
# LOAD GEOGRAPHIC ARTIFACTS
# ============================================================

load_geo_artifacts <- function(dir = ".") {
  tract <- readRDS(
    file.path(dir, "tract_acs_lookup.rds")
  )

  grocery <- readRDS(
    file.path(dir, "grocery_stores_complete.rds")
  )

  coast <- readRDS(
    file.path(dir, "coastline_raw.rds")
  )

  # Standardize tract identifiers as character values so
  # comparisons do not lose leading zeroes.

  tract$tract_fips <- as.character(
    tract$tract_fips
  )

  # Ensure the grocery and coastline spatial layers use the
  # same projected coordinate system for distance calculations.

  if (is.na(st_crs(grocery))) {
    stop(
      "grocery_stores_complete.rds has no coordinate reference system"
    )
  }

  if (st_crs(grocery)$epsg != CRS_PROJ) {
    grocery <- st_transform(
      grocery,
      CRS_PROJ
    )
  }

  if (is.na(st_crs(coast))) {
    stop(
      "coastline_raw.rds has no coordinate reference system"
    )
  }

  coast <- st_transform(
    coast,
    CRS_PROJ
  )

  acs_columns <- c(
    "pop_density",
    "poverty_rate",
    "bachelors_plus_rate",
    "owner_occ_rate"
  )

  missing_acs_columns <- setdiff(
    acs_columns,
    names(tract)
  )

  if (length(missing_acs_columns) > 0) {
    stop(
      paste0(
        "tract_acs_lookup.rds is missing columns: ",
        paste(missing_acs_columns, collapse = ", ")
      )
    )
  }

  tract_medians <- setNames(
    lapply(
      acs_columns,
      function(variable_name) {
        median(
          tract[[variable_name]],
          na.rm = TRUE
        )
      }
    ),
    acs_columns
  )

  list(
    tract = tract,
    grocery = grocery,
    coast = coast,
    tract_median = tract_medians,
    acs_cols = acs_columns
  )
}


# ============================================================
# CENSUS ADDRESS GEOCODING
# ============================================================

# Uses a one-line street address to obtain:
#
#   latitude
#   longitude
#   2020 Census tract GEOID
#
# Returns NULL if the address cannot be matched.

geocode_address <- function(address) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("httr is required for Census geocoding")
  }

  has_address <- (
    !is.null(address) &&
    length(address) == 1 &&
    !is.na(address) &&
    nzchar(trimws(address))
  )

  if (!has_address) {
    return(NULL)
  }

  url <- paste0(
    "https://geocoding.geo.census.gov/",
    "geocoder/geographies/onelineaddress"
  )

  response <- tryCatch(
    httr::GET(
      url,
      query = list(
        address = trimws(address),
        benchmark = "Public_AR_Current",
        vintage = "Census2020_Current",
        format = "json"
      ),
      httr::accept_json(),
      httr::user_agent(
        "Maine-Housing-Analytics-Engine/2.1"
      ),
      httr::timeout(10)
    ),
    error = function(e) {
      NULL
    }
  )

  if (
    is.null(response) ||
    httr::http_error(response)
  ) {
    return(NULL)
  }

  parsed <- tryCatch(
    httr::content(
      response,
      as = "parsed",
      type = "application/json"
    ),
    error = function(e) {
      NULL
    }
  )

  if (is.null(parsed)) {
    return(NULL)
  }

  matches <- parsed$result$addressMatches

  if (
    is.null(matches) ||
    length(matches) == 0
  ) {
    return(NULL)
  }

  first_match <- matches[[1]]

  latitude <- suppressWarnings(
    as.numeric(first_match$coordinates$y)
  )

  longitude <- suppressWarnings(
    as.numeric(first_match$coordinates$x)
  )

  if (
    !is_valid_latitude(latitude) ||
    !is_valid_longitude(longitude)
  ) {
    return(NULL)
  }

  tracts <- first_match$geographies[
    ["Census Tracts"]
  ]

  if (
    is.null(tracts) ||
    length(tracts) == 0
  ) {
    tract_fips <- NA_character_
  } else {
    tract_fips <- as.character(
      tracts[[1]]$GEOID
    )

    if (!is_valid_tract_fips(tract_fips)) {
      tract_fips <- NA_character_
    }
  }

  list(
    lat = latitude,
    lon = longitude,
    tract_fips = tract_fips
  )
}


# ============================================================
# CENSUS TRACT LOOKUP FROM COORDINATES
# ============================================================

# Uses existing coordinates to identify the 2020 census tract.
#
# Unlike geocode_address(), this function does not submit or
# attempt to resolve a street address.
#
# Returns an 11-digit tract GEOID or NA_character_.

lookup_tract_by_coordinates <- function(lat, lon) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop(
      "httr is required for Census coordinate lookup"
    )
  }

  if (
    !is_valid_latitude(lat) ||
    !is_valid_longitude(lon)
  ) {
    return(NA_character_)
  }

  latitude <- as.numeric(lat)
  longitude <- as.numeric(lon)

  url <- paste0(
    "https://geocoding.geo.census.gov/",
    "geocoder/geographies/coordinates"
  )

  response <- tryCatch(
    httr::GET(
      url,
      query = list(
        x = longitude,
        y = latitude,
        benchmark = "Public_AR_Current",
        vintage = "Census2020_Current",
        format = "json"
      ),
      httr::accept_json(),
      httr::user_agent(
        "Maine-Housing-Analytics-Engine/2.1"
      ),
      httr::timeout(10)
    ),
    error = function(e) {
      NULL
    }
  )

  if (
    is.null(response) ||
    httr::http_error(response)
  ) {
    return(NA_character_)
  }

  parsed <- tryCatch(
    httr::content(
      response,
      as = "parsed",
      type = "application/json"
    ),
    error = function(e) {
      NULL
    }
  )

  if (is.null(parsed)) {
    return(NA_character_)
  }

  tracts <- parsed$result$geographies[
    ["Census Tracts"]
  ]

  if (
    is.null(tracts) ||
    length(tracts) == 0
  ) {
    return(NA_character_)
  }

  tract_fips <- as.character(
    tracts[[1]]$GEOID
  )

  if (!is_valid_tract_fips(tract_fips)) {
    return(NA_character_)
  }

  tract_fips
}


# ============================================================
# DISTANCE CALCULATION
# ============================================================

minimum_distance_miles <- function(
    subject_point,
    destination_layer) {

  if (
    is.null(destination_layer) ||
    NROW(destination_layer) == 0
  ) {
    return(NA_real_)
  }

  distances <- tryCatch(
    as.numeric(
      st_distance(
        subject_point,
        destination_layer
      )
    ),
    error = function(e) {
      numeric(0)
    }
  )

  finite_distances <- distances[
    is.finite(distances)
  ]

  if (length(finite_distances) == 0) {
    return(NA_real_)
  }

  # International mile: 1,609.344 metres.

  min(finite_distances) / 1609.344
}


# ============================================================
# BUILD MODEL GEOGRAPHY
# ============================================================

build_geography <- function(
    lat,
    lon,
    tract_fips,
    geo) {

  if (!is_valid_latitude(lat)) {
    stop("Invalid latitude supplied to build_geography()")
  }

  if (!is_valid_longitude(lon)) {
    stop("Invalid longitude supplied to build_geography()")
  }

  latitude <- as.numeric(lat)
  longitude <- as.numeric(lon)

  # ----------------------------------------------------------
  # Find the tract-level ACS record
  # ----------------------------------------------------------

  if (is_valid_tract_fips(tract_fips)) {
    normalized_tract <- trimws(
      as.character(tract_fips)
    )

    matching_rows <- geo$tract[
      as.character(
        geo$tract$tract_fips
      ) == normalized_tract,
      ,
      drop = FALSE
    ]
  } else {
    matching_rows <- geo$tract[
      0,
      ,
      drop = FALSE
    ]
  }

  tract_matched <- nrow(matching_rows) > 0

  # Begin with statewide medians. If a tract was matched,
  # overwrite each median with its tract value when available.

  acs_values <- geo$tract_median

  if (tract_matched) {
    for (variable_name in geo$acs_cols) {
      tract_value <- matching_rows[
        [variable_name]
      ][1]

      if (
        length(tract_value) == 1 &&
        !is.na(tract_value) &&
        is.finite(as.numeric(tract_value))
      ) {
        acs_values[[variable_name]] <-
          as.numeric(tract_value)
      }
    }
  }

  # ----------------------------------------------------------
  # Construct the subject-property spatial point
  # ----------------------------------------------------------

  subject_point <- st_sfc(
    st_point(
      c(longitude, latitude)
    ),
    crs = CRS_GEO
  )

  subject_point_projected <- st_transform(
    subject_point,
    CRS_PROJ
  )

  # ----------------------------------------------------------
  # Calculate model distances
  # ----------------------------------------------------------

  distance_to_grocery <- minimum_distance_miles(
    subject_point_projected,
    geo$grocery
  )

  distance_to_coast <- minimum_distance_miles(
    subject_point_projected,
    geo$coast
  )

  if (is.na(distance_to_grocery)) {
    stop(
      "Could not calculate distance to grocery stores"
    )
  }

  if (is.na(distance_to_coast)) {
    stop(
      "Could not calculate distance to coastline"
    )
  }

  # ----------------------------------------------------------
  # Return model-ready geographic features
  # ----------------------------------------------------------

  list(
    features = data.frame(
      Geo.Lat = latitude,
      Geo.Lon = longitude,

      pop_density =
        acs_values$pop_density,

      poverty_rate =
        acs_values$poverty_rate,

      bachelors_plus_rate =
        acs_values$bachelors_plus_rate,

      owner_occ_rate =
        acs_values$owner_occ_rate,

      dist_grocery_miles =
        distance_to_grocery,

      dist_coast_miles =
        distance_to_coast,

      check.names = FALSE
    ),

    tract_matched = tract_matched
  )
}
