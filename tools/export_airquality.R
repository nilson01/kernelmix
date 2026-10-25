# Export the public dataset used by the documented Python analysis.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("supply the destination CSV path")
write.csv(datasets::airquality, args[1L], row.names = FALSE, na = "NA")
cat(R.version.string, "\n")
