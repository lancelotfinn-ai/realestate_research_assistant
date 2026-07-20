# ============================================================
# valuation.R
#
# Orchestrates a market-value estimate for a single property.
#
# Geography resolution order:
#   1. Use supplied latitude/longitude when available.
#   2. Use a supplied tract_fips when available.
#   3. Otherwise derive the tract from supplied coordinates.
#   4. Only geocode the street address when coordinates are absent.
#
# Prediction workflow:
#   geography -> model defaults -> supplied property facts
#   -> geographic features -> adjusted log-price prediction
#   -> current time/season adjustment -> price range
#
# Two ways to run:
#
#   A. Source this file once in the persistent R worker and call
#      value_property() repeatedly. This is the recommended method.
#
#   B. Command-line fallback:
#
#      Rscript valuation.R '<json>'
#
#      This reloads all model artifacts for each request.
# ============================================================

suppressMessages({
  library(dplyr)
  library(splines)
})

source("geo_features.R")


# ============================================================
# GENERAL HELPERS
# ============================================================

`%||%` <- function(a, b) {
  if (is.null(a)) b else a
}


is_valid_number <- function(x) {
  if (is.null(x) || length(x) != 1 || is.na(x)) {
    return(FALSE)
  }

  value <- suppressWarnings(as.numeric(x))

  !is.na(value) && is.finite(value)
}


normalize_tract_fips <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x[[1]])) {
    return(NA_character_)
  }

  value <- trimws(as.character(x[[1]]))

  if (!grepl("^[0-9]{11}$", value)) {
    return(NA_character_)
  }

  value
}


# ============================================================
# LOAD MODEL ARTIFACTS ONCE
# ============================================================

.APP_DIR <- Sys.getenv("APP_DIR", ".")

MODEL <- readRDS(
  file.path(.APP_DIR, "model_features_remarks_slim.rds")
)

DEFAULTS <- readRDS(
  file.path(.APP_DIR, "feature_defaults.rds")
)

TSCOEFS <- readRDS(
  file.path(.APP_DIR, "time_season_coefs.rds")
)

IMPACT <- readRDS(
  file.path(.APP_DIR, "impact_ranking.rds")
)

GEO <- load_geo_artifacts(.APP_DIR)


# ============================================================
# UNCERTAINTY PARAMETERS
# ============================================================

# Prefer a residual standard deviation stored on the deployed model.
# Otherwise use the residual SD measured from the full fitted model.

SIGMA <- MODEL$sigma %||% 0.305

# One residual standard deviation produces a base interval of
# approximately 68% before widening for imputed characteristics.

Z_BASE <- 1.0

# Controls how much the range widens when important property
# characteristics have been defaulted.

IMPUTE_K <- 1.0


# ============================================================
# PREDICTION HELPERS
# ============================================================

predict_log_adj <- function(model, newdata) {
  X <- model.matrix(
    delete.response(terms(model)),
    newdata
  )

  b <- coef(model)

  # Treat aliased coefficients as zero.
  b[is.na(b)] <- 0

  missing_columns <- setdiff(names(b), colnames(X))

  if (length(missing_columns) > 0) {
    stop(
      paste0(
        "Prediction matrix is missing model columns: ",
        paste(missing_columns, collapse = ", ")
      )
    )
  }

  as.numeric(
    X[, names(b), drop = FALSE] %*% b
  )
}


time_season_offset <- function(asof, ts) {
  years_since_reference <- as.numeric(
    asof - as.Date(ts$ref_date)
  ) / 365.25

  month_abbreviation <- format(asof, "%b")

  if (month_abbreviation == ts$baseline_month) {
    month_offset <- 0
  } else {
    coefficient_name <- paste0(
      "month_fct",
      month_abbreviation
    )

    if (coefficient_name %in% names(ts$beta_months)) {
      month_offset <- ts$beta_months[[coefficient_name]]
    } else {
      month_offset <- 0
    }
  }

  ts$beta_time_linear * years_since_reference +
    month_offset
}


# ============================================================
# GEOGRAPHY RESOLUTION
# ============================================================

resolve_geography <- function(
    address = NULL,
    geo_override = NULL) {

  # ----------------------------------------------------------
  # Option 1: Prefer supplied coordinates
  # ----------------------------------------------------------

  if (!is.null(geo_override)) {
    has_latitude <- is_valid_number(geo_override$lat)
    has_longitude <- is_valid_number(geo_override$lon)

    if (has_latitude && has_longitude) {
      latitude <- as.numeric(geo_override$lat)
      longitude <- as.numeric(geo_override$lon)

      supplied_tract <- normalize_tract_fips(
        geo_override$tract_fips
      )

      if (!is.na(supplied_tract)) {
        return(list(
          lat = latitude,
          lon = longitude,
          tract_fips = supplied_tract,
          geography_source = "supplied_coordinates_and_tract"
        ))
      }

      looked_up_tract <- lookup_tract_by_coordinates(
        lat = latitude,
        lon = longitude
      )

      looked_up_tract <- normalize_tract_fips(
        looked_up_tract
      )

      if (!is.na(looked_up_tract)) {
        return(list(
          lat = latitude,
          lon = longitude,
          tract_fips = looked_up_tract,
          geography_source =
            "supplied_coordinates_census_tract"
        ))
      }

      # Coordinates are still useful even if Census cannot
      # determine the tract. build_geography() will use the
      # statewide ACS medians when tract_fips is unavailable.

      return(list(
        lat = latitude,
        lon = longitude,
        tract_fips = NA_character_,
        geography_source =
          "supplied_coordinates_tract_unmatched"
      ))
    }
  }

  # ----------------------------------------------------------
  # Option 2: Geocode an address only when coordinates
  # were not available
  # ----------------------------------------------------------

  has_address <- (
    !is.null(address) &&
    length(address) == 1 &&
    !is.na(address) &&
    nzchar(trimws(address))
  )

  if (!has_address) {
    return(NULL)
  }

  geocoded <- geocode_address(trimws(address))

  if (is.null(geocoded)) {
    return(NULL)
  }

  geocoded$tract_fips <- normalize_tract_fips(
    geocoded$tract_fips
  )

  geocoded$geography_source <- "census_address_match"

  geocoded
}


# ============================================================
# MAIN VALUATION FUNCTION
# ============================================================

# user_input must be a named list using exact model-variable
# names, for example:
#
# list(
#   SqFt.Finished.Total = 2199,
#   `Lot.Size.Acres....` = 6,
#   Total.Baths = 4,
#   `X..Bedrooms` = 5,
#   Year.Built = 1975,
#   is_mh = 0,
#   is_condo = 0
# )
#
# geo_override may contain:
#
# list(
#   lat = 44.416855,
#   lon = -68.728069,
#   tract_fips = "23009966400"
# )
#
# tract_fips is optional. If omitted, the service will attempt
# to derive it from latitude and longitude.

value_property <- function(
    user_input = list(),
    address = NULL,
    geo_override = NULL,
    asof = Sys.Date()) {

  # ----------------------------------------------------------
  # 1. Resolve geography
  # ----------------------------------------------------------

  g <- resolve_geography(
    address = address,
    geo_override = geo_override
  )

  if (is.null(g)) {
    return(list(
      ok = FALSE,
      reason = "could_not_geocode",
      address = address,
      geography_source = "unresolved"
    ))
  }

  # ----------------------------------------------------------
  # 2. Construct model geography
  # ----------------------------------------------------------

  geography_result <- build_geography(
    lat = g$lat,
    lon = g$lon,
    tract_fips = g$tract_fips,
    geo = GEO
  )

  geographic_features <- geography_result$features

  # ----------------------------------------------------------
  # 3. Assemble complete model row
  # ----------------------------------------------------------

  newdata <- DEFAULTS

  supplied_variables <- intersect(
    names(user_input),
    names(newdata)
  )

  for (variable_name in supplied_variables) {
    newdata[[variable_name]] <-
      user_input[[variable_name]]
  }

  # Geographic features always override defaults because they
  # were calculated for the subject property.

  for (variable_name in names(geographic_features)) {
    newdata[[variable_name]] <-
      geographic_features[[variable_name]]
  }

  # ----------------------------------------------------------
  # 4. Generate time-adjusted point estimate
  # ----------------------------------------------------------

  predicted_adjusted_log_price <- predict_log_adj(
    MODEL,
    newdata
  )

  current_time_season_offset <- time_season_offset(
    asof,
    TSCOEFS
  )

  point_estimate <- exp(
    predicted_adjusted_log_price +
      current_time_season_offset
  )

  # ----------------------------------------------------------
  # 5. Calculate imputation-adjusted range
  # ----------------------------------------------------------

  defaultable_variables <- IMPACT$variable[
    IMPACT$is_defaulted
  ]

  imputed_variables <- setdiff(
    defaultable_variables,
    supplied_variables
  )

  imputed_impact_share <- sum(
    IMPACT$impact_share[
      IMPACT$variable %in% imputed_variables
    ],
    na.rm = TRUE
  )

  half_width <- (
    Z_BASE *
      SIGMA *
      (1 + IMPUTE_K * imputed_impact_share)
  )

  range_low <- exp(
    predicted_adjusted_log_price +
      current_time_season_offset -
      half_width
  )

  range_high <- exp(
    predicted_adjusted_log_price +
      current_time_season_offset +
      half_width
  )

  # ----------------------------------------------------------
  # 6. Identify useful follow-up questions
  # ----------------------------------------------------------

  suggested_questions <- IMPACT %>%
    filter(
      is_defaulted,
      !variable %in% supplied_variables
    ) %>%
    arrange(impact_rank) %>%
    head(4) %>%
    pull(variable)

  # ----------------------------------------------------------
  # 7. Return valuation and diagnostics
  # ----------------------------------------------------------

  list(
    ok = TRUE,

    estimate = round(point_estimate),
    low = round(range_low),
    high = round(range_high),

    asof = as.character(asof),

    n_provided = length(supplied_variables),

    imputed_impact_share = round(
      imputed_impact_share,
      3
    ),

    tract_matched = geography_result$tract_matched,

    geography_source = g$geography_source,
    latitude = g$lat,
    longitude = g$lon,
    tract_fips = g$tract_fips,

    suggest_asking_about = suggested_questions
  )
}


# ============================================================
# COMMAND-LINE FALLBACK
# ============================================================

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) >= 1 && nzchar(args[1])) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("jsonlite is required")
    }

    request <- jsonlite::fromJSON(
      args[1],
      simplifyVector = FALSE
    )

    result <- value_property(
      user_input = request$user_input %||% list(),
      address = request$address,
      geo_override = request$geo_override
    )

    cat(
      jsonlite::toJSON(
        result,
        auto_unbox = TRUE,
        null = "null"
      )
    )
  }
}
