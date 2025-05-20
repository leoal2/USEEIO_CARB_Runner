# Load required package
library(stateior)

# Define model spec, years, version, and output directory
model_spec <- "StateIOv1.3-pecan"
years <- 2012:2022
version <- utils::packageDescription("stateior", fields = "Version")
output_dir <- "C:/Users/lguillot/AppData/Local/stateio" # to change accordingly


# Load model specs from YAML
configpath <- system.file("extdata/modelspecs", paste0(model_spec, ".yml"), package = "stateior")
specs <- configr::read.config(configpath)

# Extract alias from model name (e.g., "pecan")
alias <- gsub("^.*-", "", model_spec)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Loop over all years and build/save models
for (year in years) {
  message("Building StateIO models for year: ", year)

  # Build State and TwoRegion models
  StateSupplyModel <- buildStateSupplyModel(year, specs = specs)
  StateUseModel <- buildStateUseModel(year, specs = specs)
  TwoRegionModel <- assembleTwoRegionIO(year, iolevel = specs$BaseIOLevel, specs = specs)

  # Save State Supply
  for (name in names(StateSupplyModel)) {
    df <- StateSupplyModel[[name]]
    data_name <- paste("State_Summary", name, year, version, sep = "_")
    saveRDS(df, file = file.path(output_dir, paste0(data_name, ".rds")))
    useeior:::writeMetadatatoJSON("stateior", data_name, year, "stateior", NULL, as.character(Sys.Date()), NULL)
  }

  # Save State Use
  for (name in names(StateUseModel)) {
    df <- StateUseModel[[name]]
    data_name <- paste("State_Summary", name, year, version, sep = "_")
    saveRDS(df, file = file.path(output_dir, paste0(data_name, ".rds")))
    useeior:::writeMetadatatoJSON("stateior", data_name, year, "stateior", NULL, as.character(Sys.Date()), NULL)
  }

  # Save TwoRegion
  for (name in names(TwoRegionModel)) {
    df <- TwoRegionModel[[name]]
    data_name <- paste("TwoRegion_Summary", name, alias, year, version, sep = "_")
    data_name <- gsub("_pecan", "", data_name) # to remove if needed
    saveRDS(df, file = file.path(output_dir, paste0(data_name, ".rds")))
    useeior:::writeMetadatatoJSON("stateior", data_name, year, "stateior", NULL, as.character(Sys.Date()), NULL)
  }

  message("Finished year: ", year)
}

message("All StateIO data saved in: ", output_dir)
