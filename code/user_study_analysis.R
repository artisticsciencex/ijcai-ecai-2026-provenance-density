#!/usr/bin/env Rscript

# Reproduce the Section 5 mixed-effects analysis from the released long-form data.
suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(readr)
  library(dplyr)
})

required_versions <- c(
  lme4 = "2.0-6", emmeans = "2.0.4", readr = "2.2.0", dplyr = "1.2.1"
)
actual_versions <- vapply(
  names(required_versions), function(pkg) as.character(packageVersion(pkg)), character(1)
)
if (!identical(unname(actual_versions), unname(required_versions))) {
  stop("Run Rscript code/install_r_dependencies.R to restore pinned R packages.")
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) {
  stop("Run this file with Rscript code/user_study_analysis.R")
}
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- dirname(dirname(script_path))
input_path <- file.path(repo_root, "data", "user_study_data.csv")
results_dir <- file.path(repo_root, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

study <- read_csv(input_path, show_col_types = FALSE)
expected <- c(
  "participant_id", "group", "topic", "interface", "veracity",
  "rating", "prior_knowledge"
)
if (!identical(names(study), expected)) {
  stop("Unexpected user-study schema. Expected: ", paste(expected, collapse = ", "))
}
if (n_distinct(study$participant_id) != 81 || nrow(study) != 243) {
  stop("Expected 81 participants and 243 observations.")
}

study <- study %>%
  mutate(
    participant_id = factor(participant_id),
    interface = factor(interface, levels = c("Control", "Binary", "PDI")),
    veracity = factor(veracity, levels = c("Hallucinated", "True"))
  )

# Paper specification: rating ~ Interface * Veracity + (1 | participant).
full_model <- lmer(
  rating ~ interface * veracity + (1 | participant_id),
  data = study,
  REML = FALSE
)
additive_model <- lmer(
  rating ~ interface + veracity + (1 | participant_id),
  data = study,
  REML = FALSE
)
interaction_lrt <- anova(additive_model, full_model, test = "Chisq")

cell_emmeans <- emmeans(full_model, ~ interface * veracity)
tukey_pairs <- pairs(cell_emmeans, adjust = "tukey")

cell_summary <- study %>%
  group_by(interface, veracity) %>%
  summarise(
    n = n(),
    mean = mean(rating),
    sd = sd(rating),
    se = sd / sqrt(n),
    ci95_low = mean - qt(0.975, n - 1) * se,
    ci95_high = mean + qt(0.975, n - 1) * se,
    .groups = "drop"
  )

discernment <- study %>%
  group_by(interface) %>%
  summarise(
    true_mean = mean(rating[veracity == "True"]),
    hallucinated_mean = mean(rating[veracity == "Hallucinated"]),
    pooled_sd = sqrt(
      ((sum(veracity == "True") - 1) * var(rating[veracity == "True"]) +
       (sum(veracity == "Hallucinated") - 1) * var(rating[veracity == "Hallucinated"])) /
      (sum(veracity == "True") + sum(veracity == "Hallucinated") - 2)
    ),
    cohen_d = (true_mean - hallucinated_mean) / pooled_sd,
    .groups = "drop"
  )

lrt_row <- tibble(
  comparison = "interface_by_veracity_interaction",
  chi_square = interaction_lrt$Chisq[2],
  df = interaction_lrt$`Chi Df`[2],
  p_value = interaction_lrt$`Pr(>Chisq)`[2]
)

write_csv(cell_summary, file.path(results_dir, "user_study_cell_summary.csv"))
write_csv(as.data.frame(cell_emmeans), file.path(results_dir, "user_study_emmeans.csv"))
write_csv(as.data.frame(tukey_pairs), file.path(results_dir, "user_study_tukey.csv"))
write_csv(lrt_row, file.path(results_dir, "user_study_lrt.csv"))
write_csv(discernment, file.path(results_dir, "user_study_discernment.csv"))
saveRDS(full_model, file.path(results_dir, "user_study_lmm.rds"))

print(lrt_row)
print(discernment)
