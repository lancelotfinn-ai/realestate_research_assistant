# ============================================================
# valuation.R
#
# Orchestrates a market-value estimate for a single property:
#   address  -> geography (geo_features.R)
#   defaults <- user-supplied details                (impute the rest)
#   slim model -> log_price_adj                       (predict)
#   + time/season offset for today -> price           (de-adjust)
#   band widened by how much was imputed              (honest uncertainty)
#
# Two ways to run:
#   (a) Sourced once in a long-running process (plumber): artifacts load
#       once; call value_property() per request. RECOMMENDED for speed.
#   (b) CLI shim:  Rscript valuation.R '<json>'  -> prints JSON.
#       Simple, but reloads sf + all artifacts every call (~seconds). Fine
#       to start; move to (a) when latency matters.
# ============================================================

suppressMessages({ library(dplyr); library(splines) })
source("geo_features.R")

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- load artifacts ONCE (at source time) -------------------------------
.APP_DIR <- Sys.getenv("APP_DIR", ".")
MODEL    <- readRDS(file.path(.APP_DIR, "model_features_remarks_slim.rds"))
DEFAULTS <- readRDS(file.path(.APP_DIR, "feature_defaults.rds"))
TSCOEFS  <- readRDS(file.path(.APP_DIR, "time_season_coefs.rds"))
IMPACT   <- readRDS(file.path(.APP_DIR, "impact_ranking.rds"))
GEO      <- load_geo_artifacts(.APP_DIR)

# Residual SD (log scale) for the band. Prefer one carried on the model;
# else use the constant measured from the full fit (summary(m)$sigma).
# To make it permanent, add to 06b:  m_slim$sigma <- summary(m_full)$sigma
SIGMA    <- MODEL$sigma %||% 0.305
Z_BASE   <- 1.0    # base half-width in SDs (~68% interval); lower = tighter
IMPUTE_K <- 1.0    # how much the band widens as more is imputed

# ---- prediction helpers -------------------------------------------------
predict_log_adj <- function(model, newdata) {
  X <- model.matrix(delete.response(terms(model)), newdata)
  b <- coef(model); b[is.na(b)] <- 0
  as.numeric(X[, names(b), drop = FALSE] %*% b)
}

time_season_offset <- function(asof, ts) {
  yrs  <- as.numeric(asof - as.Date(ts$ref_date)) / 365.25
  mo   <- format(asof, "%b")
  moff <- if (mo == ts$baseline_month) 0
          else (ts$beta_months[[paste0("month_fct", mo)]] %||% 0)
  ts$beta_time_linear * yrs + moff
}

# ---- main entry point ---------------------------------------------------
# user_input: named list using EXACT model variable names, e.g.
#   list(SqFt.Finished.Total = 1194, `Lot.Size.Acres....` = 3.28,
#        Total.Baths = 2, `X..Bedrooms` = 2, Year.Built = 2001, is_mh = 1, ...)
# Pass EITHER address (will be geocoded) OR geo_override = list(lat,lon,tract_fips).
value_property <- function(user_input = list(), address = NULL,
                           geo_override = NULL, asof = Sys.Date()) {
  # 1. geography
  g <- geo_override %||% geocode_address(address)
  if (is.null(g)) return(list(ok = FALSE, reason = "could_not_geocode",
                              address = address))
  gb    <- build_geography(g$lat, g$lon, g$tract_fips, GEO)
  gfeat <- gb$features

  # 2. assemble newdata:  defaults  <-overwrite-  user_input  <-then-  geography
  nd <- DEFAULTS
  provided <- intersect(names(user_input), names(nd))
  for (k in provided)        nd[[k]] <- user_input[[k]]
  for (k in names(gfeat))    nd[[k]] <- gfeat[[k]]

  # 3. predict + de-adjust to a today-dollar point estimate
  padj  <- predict_log_adj(MODEL, nd)
  off   <- time_season_offset(asof, TSCOEFS)
  point <- exp(padj + off)

  # 4. band: widen by the impact-share of what we had to impute
  defaultable   <- IMPACT$variable[IMPACT$is_defaulted]
  imputed       <- setdiff(defaultable, provided)
  imputed_share <- sum(IMPACT$impact_share[IMPACT$variable %in% imputed])
  hw            <- Z_BASE * SIGMA * (1 + IMPUTE_K * imputed_share)
  low  <- exp(padj + off - hw)
  high <- exp(padj + off + hw)

  # 5. which missing high-impact fields are worth a follow-up question
  ask <- IMPACT %>% filter(is_defaulted, !variable %in% provided) %>%
         arrange(impact_rank) %>% head(4) %>% pull(variable)

  list(ok = TRUE,
       estimate = round(point), low = round(low), high = round(high),
       asof = as.character(asof),
       n_provided = length(provided),
       imputed_impact_share = round(imputed_share, 3),
       tract_matched = gb$tract_matched,
       suggest_asking_about = ask)
}

# ---- CLI shim (only runs when called as `Rscript valuation.R '<json>'`) --
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1 && nzchar(args[1])) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite required for CLI")
    req <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
    out <- value_property(user_input  = req$user_input %||% list(),
                          address      = req$address,
                          geo_override = req$geo_override)
    cat(jsonlite::toJSON(out, auto_unbox = TRUE))
  }
}
