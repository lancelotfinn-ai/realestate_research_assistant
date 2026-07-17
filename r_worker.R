# ============================================================
# r_worker.R — long-running valuation worker.
#
# Sources valuation.R ONCE (loading sf + all model artifacts and reprojecting
# the coastline), then serves one valuation per line on stdin, replying with
# one JSON line per request on stdout. This removes the per-call artifact
# reload that made the CLI shim slow.
#
# Protocol (newline-delimited JSON):
#   in : {"id": <int>, "address": <str>, "user_input": { ... }}
#   out: {"id": <int>, "ok": true/false, ...}            (value_property output)
# Emits {"ready":true} once, after artifacts finish loading, so the parent
# knows it can start sending requests.
#
# A malformed request yields an error reply but never kills the worker. When
# stdin closes (parent gone) the loop exits cleanly.
# ============================================================

suppressMessages(library(jsonlite))

# Sourcing valuation.R loads the model, defaults, time/season coefs, impact
# ranking and all geo artifacts ONCE (see the load block at the top of that
# file). `%||%` and value_property() come from it too.
source("valuation.R")

con <- file("stdin", open = "r", blocking = TRUE)

cat('{"ready":true}\n'); flush(stdout())

repeat {
  line <- readLines(con, n = 1, warn = FALSE)
  if (length(line) == 0) break          # stdin closed -> exit
  line <- trimws(line)
  if (!nzchar(line)) next

  out <- tryCatch({
    req <- fromJSON(line, simplifyVector = FALSE)
    res <- value_property(
      user_input   = req$user_input %||% list(),
      address      = req$address,
      geo_override = req$geo_override
    )
    res$id <- req$id
    res
  }, error = function(e) {
    rid <- tryCatch(fromJSON(line, simplifyVector = FALSE)$id,
                    error = function(...) NULL)
    list(id = rid, ok = FALSE,
         reason = paste0("worker_error: ", conditionMessage(e)))
  })

  cat(toJSON(out, auto_unbox = TRUE, null = "null"), "\n", sep = "")
  flush(stdout())
}
