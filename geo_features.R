# ============================================================
# geo_features.R
#
# Geography feature builder for the live valuation tool.
# Turns an address into the 8 geographic predictors the model needs:
#   Geo.Lat, Geo.Lon, pop_density, poverty_rate, bachelors_plus_rate,
#   owner_occ_rate, dist_grocery_miles, dist_coast_miles
#
# Pure functions + an artifact loader. Sourcing this file has no side
# effects beyond defining things, so it can be loaded ONCE in a long-
# running process (see note in valuation.R about serving model).
#
# Only hard dependency at source time is `sf`. The geocoder uses httr,
# loaded lazily so you don't need it unless you actually geocode.
# ============================================================

suppressMessages(library(sf))

CRS_GEO  <- 4326       # WGS84 lat/lon (what any geocoder returns)
CRS_PROJ <- 32619      # UTM 19N -- matches grocery_stores_complete.rds

# ------------------------------------------------------------
# Load the three geography artifacts once. Project the coastline to
# CRS_PROJ at load time so per-request distance calls are cheap.
# ------------------------------------------------------------
load_geo_artifacts <- function(dir = ".") {
  tract   <- readRDS(file.path(dir, "tract_acs_lookup.rds"))
  grocery <- readRDS(file.path(dir, "grocery_stores_complete.rds"))   # already CRS_PROJ
  coast   <- st_transform(readRDS(file.path(dir, "coastline_raw.rds")), CRS_PROJ)
  if (is.na(st_crs(grocery)) || st_crs(grocery)$epsg != CRS_PROJ)
    grocery <- st_transform(grocery, CRS_PROJ)

  acs_cols <- c("pop_density","poverty_rate","bachelors_plus_rate","owner_occ_rate")
  tract_median <- setNames(lapply(acs_cols, function(v) median(tract[[v]], na.rm = TRUE)),
                           acs_cols)
  list(tract = tract, grocery = grocery, coast = coast,
       tract_median = tract_median, acs_cols = acs_cols)
}

# ------------------------------------------------------------
# Geocode a one-line address via the free Census Geocoder, which returns
# BOTH coordinates and the 2020 Census tract GEOID in one call -- the same
# tract vintage the model trained on. Returns list(lat, lon, tract_fips)
# or NULL if the address can't be matched.
# (Lazy httr load; cannot be exercised offline.)
# ------------------------------------------------------------
geocode_address <- function(address) {
  if (!requireNamespace("httr", quietly = TRUE))
    stop("httr is required for geocoding")
  url  <- "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
  resp <- tryCatch(
    httr::GET(url, query = list(address = address,
                                benchmark = "Public_AR_Current",
                                vintage   = "Census2020_Current",
                                format    = "json"),
              httr::timeout(10)),
    error = function(e) NULL)
  if (is.null(resp) || httr::http_error(resp)) return(NULL)
  j <- httr::content(resp, as = "parsed", type = "application/json")
  matches <- j$result$addressMatches
  if (length(matches) == 0) return(NULL)
  m1  <- matches[[1]]
  tr  <- m1$geographies[["Census Tracts"]]
  list(lat        = m1$coordinates$y,
       lon        = m1$coordinates$x,
       tract_fips = if (length(tr)) tr[[1]]$GEOID else NA_character_)
}

# ------------------------------------------------------------
# Build the 8 geography fields for a single point. Tract ACS values come
# from the lookup by GEOID; if the tract isn't found, fall back to the
# statewide median (flagged), so a valuation never hard-fails on geography.
# ------------------------------------------------------------
build_geography <- function(lat, lon, tract_fips, geo) {
  row <- geo$tract[geo$tract$tract_fips == tract_fips, ]
  if (nrow(row) == 0) {
    acs <- geo$tract_median; tract_matched <- FALSE
  } else {
    acs <- as.list(row[1, geo$acs_cols]); tract_matched <- TRUE
  }

  pt <- st_transform(st_sfc(st_point(c(lon, lat)), crs = CRS_GEO), CRS_PROJ)
  dist_grocery <- as.numeric(min(st_distance(pt, geo$grocery))) / 1609.34
  dist_coast   <- as.numeric(min(st_distance(pt, geo$coast)))   / 1609.34

  list(
    features = data.frame(
      Geo.Lat = lat, Geo.Lon = lon,
      pop_density          = acs$pop_density,
      poverty_rate         = acs$poverty_rate,
      bachelors_plus_rate  = acs$bachelors_plus_rate,
      owner_occ_rate       = acs$owner_occ_rate,
      dist_grocery_miles   = dist_grocery,
      dist_coast_miles     = dist_coast,
      check.names = FALSE),
    tract_matched = tract_matched
  )
}
