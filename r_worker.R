# ============================================================
# r_worker.R
#
# Persistent R valuation worker.
#
# This process:
#
#   1. Loads valuation.R once.
#   2. Loads the fitted model and geographic artifacts once.
#   3. Reads one JSON request per line from standard input.
#   4. Runs value_property().
#   5. Writes one JSON response per line to standard output.
#
# Keeping this process alive avoids reloading the R model,
# coastline, grocery-store data, and ACS data for every request.
#
# Input protocol:
#
# {
#   "id": 1,
#   "address": "16 Modin Lane, Penobscot, ME 04476",
#   "geo_override": {
#     "lat": 44.416855,
#     "lon": -68.728069,
#     "tract_fips": null
#   },
#   "user_input": {
#     "SqFt.Finished.Total": 2199,
#     "Total.Baths": 4
#   }
# }
#
# geo_override is optional.
#
# If valid coordinates are supplied, valuation.R uses them and
# does not attempt Census street-address geocoding.
#
# If coordinates are supplied without tract_fips, valuation.R
# asks Census only which tract contains those coordinates.
#
# Output protocol:
#
# {
#   "id": 1,
#   "ok": true,
#   "estimate": 1000000,
#   ...
# }
#
# A malformed request or valuation error returns an error response
# without terminating the worker.
# ============================================================


# ============================================================
# DEPENDENCIES
# ============================================================

suppressMessages(
  library(jsonlite)
)


# ============================================================
# LOAD THE VALUATION ENGINE ONCE
# ============================================================

# Sourcing valuation.R also:
#
#   - sources geo_features.R,
#   - loads the fitted model,
#   - loads feature defaults,
#   - loads time and season coefficients,
#   - loads the impact ranking,
#   - loads all geographic artifacts, and
#   - defines value_property() and %||%.
#
# Nothing from valuation.R should write ordinary output to stdout
# while it is being sourced, because stdout is reserved for the
# newline-delimited JSON protocol.

source("valuation.R")


# ============================================================
# OPEN STANDARD INPUT
# ============================================================

input_connection <- file(
  "stdin",
  open = "r",
  blocking = TRUE
)


# ============================================================
# SIGNAL READINESS TO PYTHON
# ============================================================

cat(
  '{"ready":true}\n'
)

flush(stdout())


# ============================================================
# PROCESS REQUESTS
# ============================================================

repeat {
  request_line <- readLines(
    input_connection,
    n = 1,
    warn = FALSE
  )

  # An empty result means the parent process closed stdin.

  if (length(request_line) == 0) {
    break
  }

  request_line <- trimws(request_line)

  # Ignore blank lines without terminating the worker.

  if (!nzchar(request_line)) {
    next
  }

  # ----------------------------------------------------------
  # Parse the request
  # ----------------------------------------------------------

  request <- tryCatch(
    fromJSON(
      request_line,
      simplifyVector = FALSE
    ),
    error = function(e) {
      structure(
        list(
          parse_error = conditionMessage(e)
        ),
        class = "worker_parse_error"
      )
    }
  )

  # ----------------------------------------------------------
  # Handle malformed JSON
  # ----------------------------------------------------------

  if (inherits(request, "worker_parse_error")) {
    response <- list(
      id = NULL,
      ok = FALSE,
      reason = paste0(
        "invalid_json: ",
        request$parse_error
      )
    )

    cat(
      toJSON(
        response,
        auto_unbox = TRUE,
        null = "null"
      ),
      "\n",
      sep = ""
    )

    flush(stdout())

    next
  }

  # Preserve the request ID so Python can associate the response
  # with the correct request.

  request_id <- request$id %||% NULL

  # ----------------------------------------------------------
  # Run the valuation
  # ----------------------------------------------------------

  response <- tryCatch(
    {
      result <- value_property(
        user_input =
          request$user_input %||% list(),

        address =
          request$address %||% NULL,

        geo_override =
          request$geo_override %||% NULL
      )

      result$id <- request_id

      result
    },
    error = function(e) {
      list(
        id = request_id,
        ok = FALSE,
        reason = paste0(
          "worker_error: ",
          conditionMessage(e)
        )
      )
    }
  )

  # ----------------------------------------------------------
  # Return one JSON object on one line
  # ----------------------------------------------------------

  cat(
    toJSON(
      response,
      auto_unbox = TRUE,
      null = "null"
    ),
    "\n",
    sep = ""
  )

  flush(stdout())
}


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

try(
  close(input_connection),
  silent = TRUE
)
