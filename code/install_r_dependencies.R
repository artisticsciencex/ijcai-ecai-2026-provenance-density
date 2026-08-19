#!/usr/bin/env Rscript

if (getRversion() < "4.1.0") {
  stop("R 4.1.0 or newer is required.")
}

required <- c(lme4 = "2.0-6", emmeans = "2.0.4", readr = "2.2.0", dplyr = "1.2.1")
installed <- vapply(
  names(required),
  function(pkg) requireNamespace(pkg, quietly = TRUE),
  logical(1)
)
wrong_version <- installed & vapply(
  names(required),
  function(pkg) as.character(packageVersion(pkg)) != required[[pkg]],
  logical(1)
)

if (any(!installed | wrong_version)) {
  needed <- names(required)[!installed | wrong_version]
  # Resolve compiled/transitive dependencies from CRAN first, then install the
  # exact direct package releases below (using the archive when necessary).
  install.packages(needed, repos = "https://cloud.r-project.org")
  for (pkg in needed) {
    if (requireNamespace(pkg, quietly = TRUE) &&
        as.character(packageVersion(pkg)) == required[[pkg]]) {
      next
    }
    version <- required[[pkg]]
    current_url <- sprintf(
      "https://cran.r-project.org/src/contrib/%s_%s.tar.gz", pkg, version
    )
    archive_url <- sprintf(
      "https://cran.r-project.org/src/contrib/Archive/%s/%s_%s.tar.gz",
      pkg, pkg, version
    )
    source_file <- tempfile(fileext = ".tar.gz")
    downloaded <- try(download.file(current_url, source_file, mode = "wb"), silent = TRUE)
    if (inherits(downloaded, "try-error")) {
      download.file(archive_url, source_file, mode = "wb")
    }
    install.packages(source_file, repos = NULL, type = "source")
    unlink(source_file)
  }
}

actual <- vapply(names(required), function(pkg) as.character(packageVersion(pkg)), character(1))
if (!identical(unname(actual), unname(required))) {
  stop(
    "R dependency versions do not match the release environment. Expected ",
    paste(names(required), required, sep = "=", collapse = ", "),
    "; got ", paste(names(actual), actual, sep = "=", collapse = ", ")
  )
}
