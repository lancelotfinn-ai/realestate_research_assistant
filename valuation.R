# ============================================================
# valuation.R
#
# Production valuation engine for one Maine residential property.
#
# The fitted model remains the sole source of the point estimate. This file
# deterministically translates a property-v1 record into the variables used
# during training, computes geographic predictors, applies the model's
# time/season adjustment, and returns both numbers and a traceable narrative.
#
# Public entry point:
#
#   value_property(
#     property_record = <property-v1 JSON decoded to an R list>,
#     asof = Sys.Date()
#   )
#
# The older user_input/address/geo_override contract is retained so the
# existing /estimate_home_value endpoint continues to work while main.py and
# r_worker.R are upgraded to pass a complete PropertyRecord.
# ============================================================

suppressMessages({
  library(dplyr)
  library(splines)
})

source("geo_features.R")

`%||%` <- function(a, b) if (is.null(a)) b else a

.APP_DIR <- Sys.getenv("APP_DIR", ".")

required_artifact <- function(filename) {
  path <- file.path(.APP_DIR, filename)
  if (!file.exists(path)) stop("Missing required model artifact: ", filename)
  path
}

MODEL <- readRDS(required_artifact("model_features_remarks_slim.rds"))
DEFAULTS <- readRDS(required_artifact("feature_defaults.rds"))
TSCOEFS <- readRDS(required_artifact("time_season_coefs.rds"))
IMPACT <- readRDS(required_artifact("impact_ranking.rds"))
GEO <- load_geo_artifacts(.APP_DIR)

# coefficients.json is explanatory metadata, not a prediction dependency.
COEFFICIENT_METADATA <- NULL
coefficients_path <- file.path(.APP_DIR, "coefficients.json")
if (file.exists(coefficients_path) && requireNamespace("jsonlite", quietly = TRUE)) {
  COEFFICIENT_METADATA <- tryCatch(
    jsonlite::fromJSON(coefficients_path, simplifyVector = FALSE),
    error = function(e) NULL
  )
}

MODEL_VARIABLES <- all.vars(delete.response(terms(MODEL)))
GEO_VARIABLES <- c(
  "Geo.Lat", "Geo.Lon", "pop_density", "poverty_rate",
  "bachelors_plus_rate", "owner_occ_rate",
  "dist_grocery_miles", "dist_coast_miles"
)

# A calibrated empirical interval was not exported with the model. MODEL$sigma
# is used when present; 0.305 is the residual SD recorded for this fitted model.
# The returned range is therefore labelled heuristic rather than a confidence
# or prediction interval.
SIGMA <- MODEL$sigma %||% 0.305
Z_BASE <- 1.0
IMPUTE_K <- 1.0


# ============================================================
# SAFE LIST / PROPERTY-RECORD ACCESS
# ============================================================

is_valid_number <- function(x) {
  length(x) == 1 && !is.null(x) && !is.na(x) &&
    is.finite(suppressWarnings(as.numeric(x)))
}

normalize_tract_fips <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x[[1]])) return(NA_character_)
  value <- trimws(as.character(x[[1]]))
  if (!grepl("^[0-9]{11}$", value)) return(NA_character_)
  value
}

get_path <- function(x, path, default = NULL) {
  current <- x
  for (name in path) {
    if (is.null(current) || !is.list(current) || is.null(current[[name]])) {
      return(default)
    }
    current <- current[[name]]
  }
  current
}

known_value <- function(record, path) {
  fact <- get_path(record, path)
  if (!is.list(fact) || !identical(fact$status, "known")) return(NULL)
  fact$value %||% NULL
}

observed_set <- function(record, path) {
  value <- get_path(record, path, list())
  list(
    values = as.character(value$values %||% character()),
    complete = isTRUE(value$complete)
  )
}

fact_display <- function(value) {
  if (is.logical(value)) return(if (isTRUE(value)) "yes" else "no")
  as.character(value)
}


# ============================================================
# PROPERTY-RECORD -> TRAINING VARIABLE ENCODING
# ============================================================

encode_property_record <- function(record) {
  if (!is.list(record)) stop("property_record must be a decoded JSON object")
  schema_version <- record$schema_version %||% ""
  if (!identical(schema_version, "property-v1")) {
    stop("Unsupported property_record schema_version: ", schema_version)
  }

  values <- list()
  provenance <- list()

  put <- function(variable, value, label, section) {
    if (!variable %in% MODEL_VARIABLES || is.null(value) || length(value) != 1) {
      return(invisible(NULL))
    }
    # Multiple canonical values can map to one trained dummy (for example,
    # wood/clapboard/shingle -> feat_ext_wood). Treat those mappings as OR so
    # an absent alias cannot overwrite a present alias.
    prior <- values[[variable]]
    if (!is.null(prior) && is.numeric(prior) && is.numeric(value) &&
        prior %in% c(0, 1) && value %in% c(0, 1)) {
      value <- max(prior, value)
      if (value == prior) return(invisible(NULL))
    }
    values[[variable]] <<- value
    provenance[[variable]] <<- list(label = label, section = section,
                                    value = fact_display(value))
  }

  put_fact <- function(variable, path, label, section, transform = identity) {
    value <- known_value(record, path)
    if (!is.null(value)) put(variable, transform(value), label, section)
  }

  put_bool <- function(variable, path, label, section) {
    put_fact(variable, path, label, section,
             function(x) if (isTRUE(x)) 1L else 0L)
  }

  put_set <- function(path, mapping, section) {
    observed <- observed_set(record, path)
    for (member in names(mapping)) {
      variable <- mapping[[member]][[1]]
      label <- mapping[[member]][[2]]
      if (member %in% observed$values) {
        put(variable, 1L, label, section)
      } else if (observed$complete) {
        put(variable, 0L, label, section)
      }
    }
  }

  # Core structure.
  # Core structure.
  put_fact("SqFt.Finished.Total", c("structure", "finished_square_feet"),
           "finished living area", "structure", as.numeric)
  put_fact("Lot.Size.Acres....", c("structure", "lot_acres"),
           "lot size", "structure", as.numeric)
  put_fact("X..Bedrooms", c("structure", "bedrooms"),
           "bedrooms", "structure", as.numeric)

  # Prefer an explicitly reported total bathroom count. When the MLS reports
  # full and half bathrooms separately, calculate the model input as:
  #
  #   total bathrooms = full bathrooms + 0.5 * half bathrooms
  #
  # Require both component counts so that an unknown half-bath count is not
  # silently treated as zero.
  total_bathrooms <- known_value(
    record,
    c("structure", "total_bathrooms")
  )

  if (!is.null(total_bathrooms)) {
    put(
      "Total.Baths",
      as.numeric(total_bathrooms),
      "bathrooms",
      "structure"
    )
  } else {
    full_bathrooms <- known_value(
      record,
      c("structure", "full_bathrooms")
    )

    half_bathrooms <- known_value(
      record,
      c("structure", "half_bathrooms")
    )

    if (
      !is.null(full_bathrooms) &&
      !is.null(half_bathrooms)
    ) {
      calculated_bathrooms <-
        as.numeric(full_bathrooms) +
        0.5 * as.numeric(half_bathrooms)

      put(
        "Total.Baths",
        calculated_bathrooms,
        "bathrooms calculated from full and half baths",
        "structure"
      )
    }
  }

  put_fact("Year.Built", c("structure", "year_built"),
           "year built", "structure", as.numeric)
  put_bool("feat_new_construction", c("structure", "new_construction"),
           "new construction", "structure")
           
  property_type <- known_value(record, c("structure", "property_type"))
  if (!is.null(property_type)) {
    put("is_mh", as.integer(property_type == "mobile_manufactured"),
        "manufactured/mobile home", "structure")
    put("is_condo", as.integer(property_type == "condominium"),
        "condominium ownership", "structure")
  }

  # Water and recreational access.
  put_bool("feat_water_frontage", c("water", "direct_frontage"),
           "direct water frontage", "water")
  put_bool("feat_water_view", c("water", "water_view"),
           "water view", "water")
  put_bool("feat_water_view_seasonal", c("water", "seasonal_water_view"),
           "seasonal water view", "water")
  put_set(c("water", "access_types"), list(
    deeded = c("feat_recwater_deeded", "deeded water access"),
    right_of_way = c("feat_recwater_row", "right-of-way water access"),
    nearby = c("feat_recwater_nearby", "nearby recreational water"),
    oceanfront = c("feat_recwater_oceanfront", "oceanfront setting"),
    dock = c("feat_recwater_dock", "dock access")
  ), "water")

  # Heating, cooling, and fuel.
  put_set(c("hvac", "heating_systems"), list(
    hot_water = c("feat_heat_hotwater", "hot-water heat"),
    forced_air = c("feat_heat_forcedair", "forced-air heat"),
    wood_stove = c("feat_heat_woodstove", "wood stove"),
    radiant = c("feat_heat_radiant", "radiant heat")
  ), "systems")
  put_set(c("hvac", "heating_fuels"), list(
    natural_gas = c("feat_fuel_gas_natural", "natural-gas fuel"),
    pellets = c("feat_fuel_pellets", "pellet fuel")
  ), "systems")
  put_set(c("hvac", "cooling_systems"), list(
    heat_pump = c("feat_cooling_heatpump", "heat-pump cooling"),
    central_air = c("feat_cooling_central", "central air"),
    window_units = c("feat_cooling_window", "window air conditioning")
  ), "systems")

  # Basement and foundation.
  put_set(c("basement_foundation", "basement_features"), list(
    dirt_floor = c("feat_basement_dirt", "dirt-floor basement"),
    sump_pump = c("feat_basement_sumppump", "basement sump pump")
  ), "basement_foundation")
  basement <- observed_set(record, c("basement_foundation", "basement_features"))
  basement_quality <- NULL
  if ("finished" %in% basement$values && "walkout" %in% basement$values) {
    basement_quality <- 5L
  } else if ("finished" %in% basement$values && "daylight" %in% basement$values) {
    basement_quality <- 4L
  } else if ("finished" %in% basement$values) {
    basement_quality <- 3L
  } else if ("full" %in% basement$values) {
    basement_quality <- 2L
  } else if ("crawl_space" %in% basement$values) {
    basement_quality <- 1L
  } else if (basement$complete) {
    basement_quality <- NA_integer_
  }
  if (!is.null(basement_quality)) {
    put("feat_basement_quality", basement_quality,
        "basement configuration", "basement_foundation")
  }
  put_set(c("basement_foundation", "foundation_types"), list(
    stone = c("feat_found_stone", "stone foundation"),
    fieldstone = c("feat_found_stone", "fieldstone foundation"),
    concrete_block = c("feat_found_block", "concrete-block foundation"),
    pier = c("feat_found_pier", "pier foundation"),
    slab = c("feat_found_slab", "slab foundation")
  ), "basement_foundation")

  # Interior, exterior, garage, utilities, site, and access.
  put_set(c("interior", "countertop_materials"), list(
    granite = c("feat_kitchen_granite", "granite countertops"),
    quartz = c("feat_kitchen_quartz", "quartz countertops")
  ), "interior")
  put_set(c("interior", "flooring_materials"), list(
    carpet = c("feat_floors_carpet", "carpet flooring"),
    vinyl = c("feat_floors_vinyl", "vinyl flooring"),
    luxury_vinyl = c("feat_floors_vinyl", "luxury-vinyl flooring"),
    laminate = c("feat_floors_laminate", "laminate flooring"),
    linoleum = c("feat_floors_linoleum", "linoleum flooring")
  ), "interior")
  for (spec in list(
    c("feat_kitchen_island", "kitchen_island", "kitchen island"),
    c("feat_kitchen_eatin", "eat_in_kitchen", "eat-in kitchen"),
    c("feat_primary_bath", "primary_bedroom_with_bath", "primary bedroom bath"),
    c("feat_laundry_1st", "first_floor_laundry", "first-floor laundry"),
    c("feat_inlaw_apt", "in_law_apartment", "in-law apartment")
  )) put_bool(spec[[1]], c("interior", spec[[2]]), spec[[3]], "interior")

  put_set(c("exterior", "architectural_styles"), list(
    cape_cod = c("feat_style_cape", "Cape style"),
    colonial = c("feat_style_colonial", "Colonial style"),
    contemporary = c("feat_style_contemporary", "Contemporary style"),
    new_englander = c("feat_style_newenglander", "New Englander style"),
    cottage = c("feat_style_cottage", "Cottage style"),
    farmhouse = c("feat_style_farmhouse", "Farmhouse style"),
    camp = c("feat_style_camp", "Camp style"),
    raised_ranch = c("feat_style_raised_ranch", "Raised Ranch style")
  ), "exterior")
  put_set(c("exterior", "exterior_materials"), list(
    wood = c("feat_ext_wood", "wood exterior"),
    clapboard = c("feat_ext_wood", "clapboard exterior"),
    shingle = c("feat_ext_wood", "wood-shingle exterior"),
    brick = c("feat_ext_brick", "brick exterior"),
    log = c("feat_ext_log", "log exterior"),
    asbestos = c("feat_ext_asbestos", "asbestos exterior"),
    fiber_cement = c("feat_ext_fibcement", "fiber-cement exterior")
  ), "exterior")
  put_set(c("exterior", "roof_materials"), list(
    metal = c("feat_roof_metal", "metal roof"),
    flat = c("feat_roof_flat", "flat roof")
  ), "exterior")
  for (spec in list(
    c("feat_deck", "deck", "deck"),
    c("feat_porch_screened", "screened_porch", "screened porch"),
    c("feat_pool_inground", "in_ground_pool", "in-ground pool"),
    c("feat_barn", "barn", "barn")
  )) put_bool(spec[[1]], c("exterior", spec[[2]]), spec[[3]], "exterior")

  for (spec in list(
    c("feat_garage_attached", "attached", "attached garage"),
    c("feat_garage_directentry", "direct_entry", "direct-entry garage"),
    c("feat_garage_heated", "heated", "heated garage")
  )) put_bool(spec[[1]], c("garage", spec[[2]]), spec[[3]], "garage")

  for (spec in list(
    c("feat_water_public", "public_water", "public water"),
    c("feat_sewer_public", "public_sewer", "public sewer"),
    c("feat_fuel_gas_natural", "natural_gas_available", "natural gas"),
    c("feat_generator", "generator", "generator"),
    c("feat_radon_air", "radon_air_mitigation", "radon-air mitigation"),
    c("feat_solar", "seller_owned_solar", "seller-owned solar"),
    c("feat_double_pane", "double_pane_windows", "double-pane windows")
  )) put_bool(spec[[1]], c("utilities_equipment", spec[[2]]), spec[[3]], "systems")

  put_set(c("access_location", "road_types"), list(
    private = c("feat_road_private", "private road"),
    dirt = c("feat_road_dirt", "dirt road"),
    seasonal = c("feat_road_seasonal", "seasonal road")
  ), "location")
  put_bool("feat_driveway_paved", c("access_location", "paved_driveway"),
           "paved driveway", "location")
  put_set(c("access_location", "location_tags"), list(
    intown = c("feat_loc_intown", "in-town location"),
    ski_resort = c("feat_loc_ski", "ski-resort location"),
    near_public_beach = c("feat_loc_beach", "near a public beach")
  ), "location")
  put_set(c("access_location", "site_tags"), list(
    wooded = c("feat_site_wooded", "wooded site"),
    cul_de_sac = c("feat_site_culdesac", "cul-de-sac site")
  ), "location")
  put_bool("feat_view_scenic", c("access_location", "scenic_view"),
           "scenic view", "location")
  put_bool("feat_view_mountain", c("access_location", "mountain_view"),
           "mountain view", "location")

  # Public-remarks classifications use the exact ordinal/dummy encodings from
  # 04_build_model_df_since_sept2025.R. Disclosure-only facts are intentionally
  # not converted into rem_* fields because those terms were trained on public
  # marketing remarks.
  remarks <- record$remarks_classification
  if (is.list(remarks)) {
    condition <- known_value(record, c("remarks_classification", "condition"))
    condition_map <- c("move-in-ready" = 5, updated = 4, dated = 3,
                       "needs-work" = 2, fixer = 1)
    if (!is.null(condition) && condition %in% names(condition_map)) {
      put("rem_condition", unname(condition_map[[condition]]),
          "remarks condition tier", "condition")
    } else if (identical(condition, "unknown")) {
      put("rem_condition", NA_integer_, "remarks condition unknown", "condition")
    }

    remarks_bools <- list(
      new_roof = c("rem_new_roof", "recent roof work"),
      new_heating = c("rem_new_heating", "recent heating work"),
      new_windows = c("rem_new_windows", "recent window work"),
      new_basement_work = c("rem_new_basement", "recent basement work"),
      systems_updated = c("rem_systems_updated", "updated major systems"),
      water_issues = c("rem_water_issues", "advertised water issue"),
      as_is_sale = c("rem_as_is", "as-is marketing"),
      estate_or_trust_sale = c("rem_estate_sale", "estate/trust sale"),
      known_defects_advertised = c("rem_known_defects", "advertised defects"),
      investor_language = c("rem_investor", "investor-oriented marketing"),
      historical_character = c("rem_historic", "historic character"),
      privacy_high = c("rem_privacy_high", "high-privacy marketing"),
      privacy_low = c("rem_privacy_low", "low-privacy/in-town marketing"),
      views_described = c("rem_views", "views described in remarks")
    )
    for (name in names(remarks_bools)) {
      value <- known_value(record, c("remarks_classification", name))
      if (!is.null(value)) put(remarks_bools[[name]][[1]], as.integer(isTRUE(value)),
                               remarks_bools[[name]][[2]], "remarks")
    }

    foundation <- known_value(record, c("remarks_classification", "foundation_signal"))
    if (!is.null(foundation)) {
      put("rem_foundation_pos", as.integer(foundation == "positive"),
          "positive foundation signal", "condition")
      put("rem_foundation_neg", as.integer(foundation == "negative"),
          "negative foundation signal", "condition")
    }

    quality_maps <- list(
      kitchen_quality = c("high-end" = 3, updated = 2, standard = 1,
                          dated = 0, poor = 0),
      bath_quality = c("high-end" = 3, updated = 2, standard = 1,
                       dated = 0, poor = 0),
      flooring_quality = c("high-end" = 3, updated = 2, standard = 1,
                           dated = 0, poor = 0)
    )
    quality_variables <- c(kitchen_quality = "rem_kitchen_quality",
                           bath_quality = "rem_bath_quality",
                           flooring_quality = "rem_flooring_quality")
    for (name in names(quality_maps)) {
      value <- known_value(record, c("remarks_classification", name))
      if (!is.null(value) && value %in% names(quality_maps[[name]])) {
        put(quality_variables[[name]], unname(quality_maps[[name]][[value]]),
            paste(gsub("_", " ", name), "tier"), "quality")
      } else if (identical(value, "unknown")) {
        put(quality_variables[[name]], NA_integer_,
            paste(gsub("_", " ", name), "unknown"), "quality")
      }
    }

    bucolic <- known_value(record, c("remarks_classification", "bucolic_character"))
    bucolic_map <- c(high = 3, moderate = 2, low = 1)
    if (!is.null(bucolic) && bucolic %in% names(bucolic_map)) {
      put("rem_bucolic", unname(bucolic_map[[bucolic]]),
          "rural/bucolic character", "location")
    } else if (identical(bucolic, "unknown")) {
      put("rem_bucolic", NA_integer_, "rural character unknown", "location")
    }

    distress <- known_value(record, c("remarks_classification", "distress"))
    if (!is.null(distress)) {
      put("rem_distress_strong", as.integer(distress == "strong"),
          "strong distress signal", "condition")
      put("rem_distress_mod", as.integer(distress == "moderate"),
          "moderate distress signal", "condition")
    }

    lifestyle <- known_value(record, c("remarks_classification", "lifestyle_tier"))
    lifestyle_variables <- c(
      luxury = "rem_lifestyle_luxury", upscale = "rem_lifestyle_upscale",
      starter = "rem_lifestyle_starter", retirement = "rem_lifestyle_retire",
      "camp-seasonal" = "rem_lifestyle_camp", investment = "rem_lifestyle_invest"
    )
    if (!is.null(lifestyle) && lifestyle != "unknown") {
      for (name in names(lifestyle_variables)) {
        put(lifestyle_variables[[name]], as.integer(lifestyle == name),
            paste(name, "lifestyle marketing"), "remarks")
      }
    }
  }

  # Disclosure facts are returned as caveats. They do not silently change a
  # coefficient trained on public remarks.
  disclosure_caveats <- character()
  caveat_fields <- list(
    active_roof_leak = "The disclosure reports an active roof leak.",
    past_roof_leak = "The disclosure reports a past roof leak.",
    basement_water_intrusion = "The disclosure reports basement water intrusion.",
    foundation_problem = "The disclosure reports a foundation problem.",
    mold_reported = "The disclosure reports mold.",
    radon_issue_reported = "The disclosure reports a radon issue.",
    septic_problem = "The disclosure reports a septic problem.",
    hazardous_material_reported = "The disclosure reports a hazardous material."
  )
  for (name in names(caveat_fields)) {
    if (isTRUE(known_value(record, c("disclosures", name)))) {
      disclosure_caveats <- c(disclosure_caveats, caveat_fields[[name]])
    }
  }
  material_defects <- get_path(
    record,
    c("disclosures", "material_defects"),
    list()
  )

  non_defect_placeholders <- c(
    "",
    "none",
    "none known",
    "no known defects",
    "no defects known",
    "no known issues",
    "not applicable",
    "n/a",
    "na"
  )

  for (fact in material_defects) {
    if (
      !is.list(fact) ||
      !identical(fact$status, "known") ||
      is.null(fact$value)
    ) {
      next
    }

    defect_text <- trimws(as.character(fact$value))
    normalized_text <- tolower(defect_text)

    if (
      !nzchar(defect_text) ||
      normalized_text %in% non_defect_placeholders
    ) {
      next
    }

    disclosure_caveats <- c(
      disclosure_caveats,
      paste0("Disclosure item: ", defect_text)
    )
  }
  
  list(values = values, provenance = provenance,
       disclosure_caveats = unique(disclosure_caveats))
}


# ============================================================
# GEOGRAPHY AND PREDICTION
# ============================================================

property_record_location <- function(record) {
  address <- known_value(record, c("address", "full_address"))
  lat <- known_value(record, c("coordinates", "latitude"))
  lon <- known_value(record, c("coordinates", "longitude"))
  list(address = address, lat = lat, lon = lon)
}

resolve_geography <- function(address = NULL, geo_override = NULL) {
  if (!is.null(geo_override) && is_valid_number(geo_override$lat) &&
      is_valid_number(geo_override$lon)) {
    lat <- as.numeric(geo_override$lat)
    lon <- as.numeric(geo_override$lon)
    tract <- normalize_tract_fips(geo_override$tract_fips)
    if (!is.na(tract)) return(list(lat = lat, lon = lon, tract_fips = tract,
                                   geography_source = "supplied_coordinates_and_tract"))
    tract <- normalize_tract_fips(lookup_tract_by_coordinates(lat, lon))
    return(list(lat = lat, lon = lon, tract_fips = tract,
                geography_source = if (is.na(tract))
                  "supplied_coordinates_tract_unmatched" else
                  "supplied_coordinates_census_tract"))
  }
  if (is.null(address) || length(address) != 1 || is.na(address) ||
      !nzchar(trimws(address))) return(NULL)
  geocoded <- geocode_address(trimws(address))
  if (is.null(geocoded)) return(NULL)
  geocoded$tract_fips <- normalize_tract_fips(geocoded$tract_fips)
  geocoded$geography_source <- "census_address_match"
  geocoded
}

predict_log_adjusted <- function(newdata) {
  X <- model.matrix(delete.response(terms(MODEL)), newdata)
  b <- coef(MODEL)
  b[is.na(b)] <- 0
  missing_columns <- setdiff(names(b), colnames(X))
  if (length(missing_columns)) {
    stop("Prediction matrix is missing model columns: ",
         paste(missing_columns, collapse = ", "))
  }
  as.numeric(X[, names(b), drop = FALSE] %*% b)
}

time_season_offset <- function(asof) {
  years <- as.numeric(asof - as.Date(TSCOEFS$ref_date)) / 365.25
  month <- format(asof, "%b")
  month_name <- paste0("month_fct", month)
  month_effect <- if (month == TSCOEFS$baseline_month) 0 else
    TSCOEFS$beta_months[[month_name]] %||% 0
  TSCOEFS$beta_time_linear * years + month_effect
}

currency <- function(x) paste0("$", format(round(x, -3), big.mark = ",",
                                            scientific = FALSE, trim = TRUE))

variable_label <- function(variable, provenance) {
  if (!is.null(provenance[[variable]]$label)) return(provenance[[variable]]$label)
  gsub("[._]+", " ", variable)
}

build_driver_analysis <- function(subject_row, baseline_row, supplied, provenance,
                                  offset, point_estimate) {
  results <- list()
  for (variable in supplied) {
    comparison <- subject_row
    comparison[[variable]] <- baseline_row[[variable]]
    comparison_log <- tryCatch(predict_log_adjusted(comparison),
                               error = function(e) NA_real_)
    if (!is.finite(comparison_log)) next
    comparison_price <- exp(comparison_log + offset)
    results[[length(results) + 1]] <- data.frame(
      variable = variable,
      label = variable_label(variable, provenance),
      dollar_effect = point_estimate - comparison_price,
      pct_effect = 100 * (point_estimate / comparison_price - 1),
      comparison = "subject value versus the training-sample default, holding other subject inputs fixed",
      stringsAsFactors = FALSE
    )
  }
  if (!length(results)) return(data.frame())
  drivers <- bind_rows(results)
  drivers[order(-abs(drivers$dollar_effect)), , drop = FALSE]
}

build_narrative <- function(estimate, low, high, drivers, caveats,
                            imputed_share, geography_result, asof) {
  opening <- paste0(
    "The model estimates market value at ", currency(estimate),
    " as of ", as.character(asof), ", with a heuristic range of ",
    currency(low), " to ", currency(high), "."
  )
  driver_text <- character()
  if (nrow(drivers)) {
    top <- head(drivers, 5)
    driver_text <- vapply(seq_len(nrow(top)), function(i) {
      direction <- if (top$dollar_effect[[i]] >= 0) "above" else "below"
      paste0(top$label[[i]], " moved the estimate approximately ",
             currency(abs(top$dollar_effect[[i]])), " ", direction,
             " the model-default comparison")
    }, character(1))
  }
  geography_text <- paste0(
    "Location inputs used actual coordinates and ",
    if (isTRUE(geography_result$tract_matched)) "matched Census-tract context."
    else "statewide-median Census context because the tract was not matched."
  )
  limitation <- paste0(
    "About ", round(100 * imputed_share),
    "% of ranked model-input impact was defaulted because those characteristics were not supplied. ",
    "The range is based on residual model spread and missing-input widening; it is not a calibrated appraisal confidence interval."
  )
  list(
    summary = opening,
    principal_drivers = driver_text,
    geography = geography_text,
    disclosure_caveats = caveats,
    limitations = limitation
  )
}


# ============================================================
# PUBLIC VALUATION FUNCTION
# ============================================================

value_property <- function(user_input = list(), address = NULL,
                           geo_override = NULL, asof = Sys.Date(),
                           property_record = NULL) {
  asof <- as.Date(asof)
  if (is.na(asof)) stop("asof must be a valid date")

  provenance <- list()
  caveats <- character()
  input_mode <- "legacy_model_fields"

  if (!is.null(property_record)) {
    encoded <- encode_property_record(property_record)
    user_input <- encoded$values
    provenance <- encoded$provenance
    caveats <- encoded$disclosure_caveats
    location <- property_record_location(property_record)
    address <- address %||% location$address
    if (is.null(geo_override) && is_valid_number(location$lat) &&
        is_valid_number(location$lon)) {
      geo_override <- list(lat = location$lat, lon = location$lon,
                           tract_fips = NULL)
    }
    input_mode <- "property-v1"
  }

  geography <- resolve_geography(address, geo_override)
  if (is.null(geography)) {
    return(list(ok = FALSE, reason = "could_not_resolve_geography",
                address = address, geography_source = "unresolved"))
  }
  geography_result <- build_geography(geography$lat, geography$lon,
                                      geography$tract_fips, GEO)

  baseline <- DEFAULTS
  subject <- DEFAULTS
  supplied <- intersect(names(user_input), names(subject))
  ignored <- setdiff(names(user_input), names(subject))
  for (variable in supplied) subject[[variable]] <- user_input[[variable]]
  for (variable in names(geography_result$features)) {
    subject[[variable]] <- geography_result$features[[variable]]
    baseline[[variable]] <- geography_result$features[[variable]]
  }

  log_adjusted <- predict_log_adjusted(subject)
  offset <- time_season_offset(asof)
  estimate <- exp(log_adjusted + offset)

  defaultable <- IMPACT$variable[IMPACT$is_defaulted]
  imputed <- setdiff(defaultable, supplied)
  imputed_share <- sum(IMPACT$impact_share[IMPACT$variable %in% imputed],
                       na.rm = TRUE)
  half_width <- Z_BASE * SIGMA * (1 + IMPUTE_K * imputed_share)
  low <- exp(log_adjusted + offset - half_width)
  high <- exp(log_adjusted + offset + half_width)

  drivers <- build_driver_analysis(subject, baseline, supplied, provenance,
                                   offset, estimate)
  suggested <- IMPACT %>%
    filter(is_defaulted, !variable %in% supplied) %>%
    arrange(impact_rank) %>% head(6) %>% pull(variable)
  narrative <- build_narrative(estimate, low, high, drivers, caveats,
                               imputed_share, geography_result, asof)

  list(
    ok = TRUE,
    estimate = round(estimate), low = round(low), high = round(high),
    range_method = "heuristic_residual_spread_with_imputation_widening",
    asof = as.character(asof), input_mode = input_mode,
    model = list(
      description = COEFFICIENT_METADATA$metadata$model_description %||%
        "Maine statewide hedonic price model",
      training_period = COEFFICIENT_METADATA$metadata$training_period %||% NULL,
      n_observations = COEFFICIENT_METADATA$metadata$n_observations %||% NULL,
      r_squared = COEFFICIENT_METADATA$metadata$r_squared %||% NULL
    ),
    n_provided = length(supplied), supplied_variables = supplied,
    ignored_variables = ignored,
    imputed_impact_share = round(imputed_share, 3),
    imputed_variables = imputed,
    geography_source = geography$geography_source,
    tract_matched = geography_result$tract_matched,
    latitude = geography$lat, longitude = geography$lon,
    tract_fips = geography$tract_fips,
    geographic_features = geography_result$features,
    drivers = if (nrow(drivers)) head(drivers, 10) else list(),
    narrative = narrative,
    suggest_asking_about = suggested
  )
}


# ============================================================
# COMMAND-LINE FALLBACK
# ============================================================

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1 && nzchar(args[[1]])) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
    request <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
    result <- value_property(
      user_input = request$user_input %||% list(),
      address = request$address %||% NULL,
      geo_override = request$geo_override %||% NULL,
      asof = request$asof %||% Sys.Date(),
      property_record = request$property_record %||% NULL
    )
    cat(jsonlite::toJSON(result, auto_unbox = TRUE, null = "null",
                        dataframe = "rows"))
  }
}