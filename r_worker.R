# ============================================================
# r_worker.R
#
# Persistent newline-delimited JSON worker for valuation.R.
#
# The worker loads the fitted model and geographic artifacts once, then
# accepts one request per line on stdin and writes one response per line on
# stdout. Ordinary diagnostics must go to stderr because stdout is reserved
# for the JSON protocol used by main.py.
#
# Supported request forms
# -----------------------
#
# 1. Full extracted property record (preferred):
#
# {
#   "id": 1,
#   "property_record": { "schema_version": "property-v1", ... },
#   "asof": "2026-07-22"
# }
#
# 2. Legacy flat model inputs (retained for /estimate_home_value):
#
# {
#   "id": 2,
#   "address": "49 Bagaduce Lane, Penobscot, ME",
#   "geo_override": {
#     "lat": 44.4056,
#     "lon": -68.699,
#     "tract_fips": "23009966400"
#   },
#   "user_input": {
#     "SqFt.Finished.Total": 2199,
#     "Total.Baths": 4
#   },
#   "asof": "2026-07-22"
# }
#
# property_record, user_input, address, geo_override, and asof are optional at
# the transport layer. valuation.R performs the substantive validation.
# ============================================================

suppressMessages(
  library(jsonlite)
)


# ============================================================
# LOAD THE VALUATION ENGINE ONCE
# ============================================================

app_dir <- Sys.getenv("APP_DIR", ".")
valuation_path <- file.path(app_dir, "valuation.R")

if (!file.exists(valuation_path)) {
  stop("Could not find valuation.R at: ", valuation_path)
}

# Sourcing valuation.R loads the model, defaults, time/season coefficients,
# impact ranking, coefficient metadata, and geographic artifacts once.
source(valuation_path)


# ============================================================
# JSON PROTOCOL HELPERS
# ============================================================

write_response <- function(response) {
  encoded <- toJSON(
    response,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    dataframe = "rows",
    digits = NA
  )

  cat(encoded, "\n", sep = "")
  flush(stdout())
}


error_response <- function(id = NULL, reason, error_type = NULL) {
  list(
    id = id,
    ok = FALSE,
    reason = reason,
    error_type = error_type
  )
}


parse_request <- function(line) {
  tryCatch(
    fromJSON(
      line,
      simplifyVector = FALSE
    ),
    error = function(e) {
      structure(
        list(message = conditionMessage(e)),
        class = "worker_parse_error"
      )
    }
  )
}


# ============================================================
# OPEN STDIN AND SIGNAL READINESS
# ============================================================

input_connection <- file(
  "stdin",
  open = "r",
  blocking = TRUE
)

# main.py waits for this message before sending its first valuation request.
write_response(list(ready = TRUE))


# ============================================================
# PROCESS REQUESTS
# ============================================================

repeat {
  request_line <- readLines(
    input_connection,
    n = 1,
    warn = FALSE
  )

  # EOF means the Python parent closed the worker's stdin.
  if (length(request_line) == 0) break

  request_line <- trimws(request_line)
  if (!nzchar(request_line)) next

  request <- parse_request(request_line)

  if (inherits(request, "worker_parse_error")) {
    write_response(
      error_response(
        id = NULL,
        reason = paste0("invalid_json: ", request$message),
        error_type = "parse_error"
      )
    )
    next
  }

  if (!is.list(request)) {
    write_response(
      error_response(
        id = NULL,
        reason = "invalid_request: top-level JSON value must be an object",
        error_type = "validation_error"
      )
    )
    next
  }

  request_id <- request$id %||% NULL

  response <- tryCatch(
    {
      result <- value_property(
        user_input = request$user_input %||% list(),
        address = request$address %||% NULL,
        geo_override = request$geo_override %||% NULL,
        asof = request$asof %||% Sys.Date(),
        property_record = request$property_record %||% NULL
      )

      if (!is.list(result)) {
        stop("value_property() did not return a list")
      }

      # Put the transport ID on the result so main.py can associate the
      # response with the request that produced it.
      result$id <- request_id
      result
    },
    error = function(e) {
      # Do not return document text, credentials, stack traces, or local paths.
      # The error class and condition message are sufficient for service logs
      # and client-side diagnosis of model/input incompatibilities.
      message(
        "[rworker] valuation failed: ",
        class(e)[[1]],
        ": ",
        conditionMessage(e)
      )

      error_response(
        id = request_id,
        reason = paste0("worker_error: ", conditionMessage(e)),
        error_type = class(e)[[1]]
      )
    }
  )

  write_response(response)
}


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

try(
  close(input_connection),
  silent = TRUE
)