rm(list=ls())
library(dplyr)
library(tidyr)

# PART 1: MERGING ALL FINE ROOT TRAIT DATA -------------------------------------
# 
# Fine root respiration ----
FR_resp_data_all = readRDS("/Users/edwardzhu/EdZhu-Drive/Projects_LPICEA_2025/25-rrgt-NEW-Code/outdata/step1-Rrdata_all_v0515.rds")
Rrdata_indv_touse = FR_resp_data_all$Rrdata_indvobs_filtered
Rrdata_indv_touse$resp_q10_Tj2001 = 3.22 - 0.046*(Rrdata_indv_touse$MeasurementTempValue)

Rrdata_indv_touse$TraitValue25_Tj2001 = Rrdata_indv_touse$TraitValueRaw * (Rrdata_indv_touse$resp_q10_Tj2001 ^ ((25 - Rrdata_indv_touse$MeasurementTempValue)/10))

FR_resp_data_sites = Rrdata_indv_touse %>%
  group_by(SiteLatitude, SiteLongitude) %>%
  summarise(CWM_Rr25_Tj2001 = mean(TraitValue25_Tj2001, na.rm = TRUE), .groups = 'drop')

rm(list=setdiff(ls(), "FR_resp_data_sites"))

# Morphological traits ----
FR_morph_data_sites = readRDS("/Users/edwardzhu/EdZhu-Drive/Projects_LPICEA_2025/25-LTWG-FD-Paper/250401-TRYdata-cleaned/FRtrait_merged_fixTRY_250401.rds")

FR_morph_data_sites = FR_morph_data_sites %>%
  select(SiteLatitude, SiteLongitude, TraitName, TraitValueSTD) %>%
  pivot_wider(names_from = TraitName, values_from = TraitValueSTD, values_fn = mean) %>%
  rename(CWM_RD = Mean_Root_diameter, CWM_SRL = Specific_root_length, CWM_RTD = Root_tissue_density)



# Exudation rate ----
FR_exudation_data_sites = read.csv("/Users/edwardzhu/Documents/ICCS_summerschool/Hackathon25-Zcost/dataset/exudate-flux-data.csv") %>% rename(SiteLatitude=Lat, SiteLongitude=Lon) %>%
  group_by(SiteLatitude, SiteLongitude) %>%
  summarise(CWM_Exudation = mean(Exudation.rate, na.rm = TRUE), .groups = 'drop')


FR_alltraits_commonsites = FR_resp_data_sites %>%
  full_join(FR_morph_data_sites, by = c("SiteLatitude", "SiteLongitude")) %>%
  full_join(FR_exudation_data_sites, by = c("SiteLatitude", "SiteLongitude"))

write.csv(FR_alltraits_commonsites, "FR_alltraits_sites_outer.csv", row.names = FALSE)
colnames(FR_alltraits_commonsites)
# [1] "SiteLatitude"    "SiteLongitude"   "CWM_Rr25_Tj2001" "CWM_RD"          "CWM_SRL"         "CWM_RTD"        
# [7] "CWM_Exudation"

# PART 2: GAP FILLING ---------------------------------------------------`
rm(list=ls())
if(!require("rstan")) install.packages("rstan")
if(!require("Matrix")) install.packages("Matrix")
if(!require("bayesplot")) install.packages("bayesplot")
library(rstan)
library(Matrix)
library(bayesplot)
library(tidyverse)

# Read in the trait dataset
FR_traits <- read.csv("FR_alltraits_sites_outer.csv")

# Prepare data for BHPMF
# Extract trait matrix and standardize values
trait_cols <- c("CWM_Rr25_Tj2001", "CWM_RD", "CWM_SRL", "CWM_RTD", "CWM_Exudation")
trait_matrix <- as.matrix(FR_traits[, trait_cols])

# Standardize non-missing values for each trait (column)
trait_means <- colMeans(trait_matrix, na.rm = TRUE)
trait_sds <- apply(trait_matrix, 2, sd, na.rm = TRUE)
trait_matrix_std <- scale(trait_matrix, center = trait_means, scale = trait_sds)

# Create observation mask (1 for observed, 0 for missing)
obs_mask <- 1 * (!is.na(trait_matrix_std))
n_sites <- nrow(trait_matrix_std)
n_traits <- ncol(trait_matrix_std)

# BHPMF Stan model
stan_code <- "
data {
  int<lower=0> n_sites;       // number of sites
  int<lower=0> n_traits;      // number of traits
  int<lower=0> n_obs;         // number of observed values
  int<lower=1, upper=n_sites> site_idx[n_obs];  // site indices for observations
  int<lower=1, upper=n_traits> trait_idx[n_obs]; // trait indices for observations
  real y_obs[n_obs];          // observed trait values (standardized)
  int<lower=1> K;             // latent dimension
}

parameters {
  matrix[n_sites, K] U;       // site latent factors
  matrix[n_traits, K] V;      // trait latent factors
  real<lower=0> sigma;        // observation noise
  vector<lower=0>[K] lambda_u; // site regularization parameters
  vector<lower=0>[K] lambda_v; // trait regularization parameters
  real<lower=0> alpha_u;      // hyperprior for lambda_u
  real<lower=0> alpha_v;      // hyperprior for lambda_v
}

model {
  alpha_u ~ gamma(1, 1);
  alpha_v ~ gamma(1, 1);

  for (k in 1:K) {
    lambda_u[k] ~ gamma(alpha_u, 1);
    lambda_v[k] ~ gamma(alpha_v, 1);

    // Regularized priors for latent factors
    for (i in 1:n_sites) U[i, k] ~ normal(0, inv_sqrt(lambda_u[k]));
    for (j in 1:n_traits) V[j, k] ~ normal(0, inv_sqrt(lambda_v[k]));
  }

  sigma ~ cauchy(0, 1);

  // Likelihood
  for (n in 1:n_obs)
    y_obs[n] ~ normal(dot_product(U[site_idx[n]], V[trait_idx[n]]), sigma);
}

generated quantities {
  matrix[n_sites, n_traits] y_pred;

  for (i in 1:n_sites)
    for (j in 1:n_traits)
      y_pred[i, j] = dot_product(U[i], V[j]);
}
"

# Prepare data for Stan
observed_indices <- which(obs_mask == 1, arr.ind = TRUE)
stan_data <- list(
  n_sites = n_sites,
  n_traits = n_traits,
  n_obs = nrow(observed_indices),
  site_idx = observed_indices[, 1],
  trait_idx = observed_indices[, 2],
  y_obs = trait_matrix_std[obs_mask == 1],
  K = 3  # Number of latent dimensions
)

# Compile and run the Stan model
set.seed(123)
bhpmf_fit <- stan(
  model_code = stan_code,
  data = stan_data,
  iter = 2000,
  chains = 4,
  cores = 4,
  warmup = 1000,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)

# Extract predicted values
y_pred_samples <- rstan::extract(bhpmf_fit, "y_pred")$y_pred
y_pred_mean <- apply(y_pred_samples, c(2, 3), mean)

# Transform predictions back to original scale
trait_matrix_filled <- y_pred_mean * matrix(rep(trait_sds, each = n_sites),
                                          nrow = n_sites) +
                      matrix(rep(trait_means, each = n_sites), nrow = n_sites)

# Replace only the missing values with predictions
for (i in 1:n_sites) {
  for (j in 1:n_traits) {
    if (is.na(trait_matrix[i, j])) {
      trait_matrix[i, j] <- trait_matrix_filled[i, j]
    }
  }
}

# Create the filled dataset
FR_traits_filled <- FR_traits
FR_traits_filled[, trait_cols] <- trait_matrix

# Save the gap-filled dataset
write.csv(FR_traits_filled, "FR_alltraits_sites_filled_BHPMF.csv", row.names = FALSE)

# Create diagnostic plots
# Plot observed vs predicted values for validation
plot_data <- data.frame(
  observed = trait_matrix_std[obs_mask == 1],
  predicted = y_pred_mean[obs_mask == 1]
)

plt1_gapfill_scatter = ggplot(plot_data, aes(x = observed, y = predicted)) +
  geom_point(alpha = 0.5, color="gray20") +
  geom_abline(intercept = 0, slope = 1, color = "darkred", linetype = "dashed") +
  annotate("text", x = 0.5, y = 1.5,
           label = paste("RMSE:", round(sqrt(mean((plot_data$observed - plot_data$predicted)^2)), 3),
                         "\nR2:", round(cor(plot_data$observed, plot_data$predicted)^2, 3)),
           size = 4, color = "black") +
  labs(title = "Observed vs Predicted, Gap Filling",
       x = "Standardized Observed Values",
       y = "Standardized Predicted Values") +
  theme_minimal()
ggsave("gapfill_scatter_plot.pdf", plt1_gapfill_scatter, width = 4.5, height = 4.5)

# Calculate performance metrics
rmse <- sqrt(mean((plot_data$observed - plot_data$predicted)^2))
r_squared <- cor(plot_data$observed, plot_data$predicted)^2

cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")

# Check imputation summary
summary_filled <- colSums(!is.na(FR_traits_filled[, trait_cols]))
summary_orig <- colSums(!is.na(FR_traits[, trait_cols]))
imputation_stats <- data.frame(
  trait = trait_cols,
  original_count = summary_orig,
  filled_count = summary_filled,
  percent_filled = round(100 * (summary_filled - summary_orig) / n_sites, 1)
)
print(imputation_stats)

write_csv(FR_traits_filled, "FR_alltraits_sites_filled_BHPMF.csv")
