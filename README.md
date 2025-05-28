This guide assumes you are using Anaconda Prompt (Windows). Please install Miniconda or Anaconda before following these steps.

# USEEIO_CARB_Runner

This repository provides a California-customized implementation of the U.S. EPA's Environmentally-Extended Input-Output (USEEIO) model. It includes a Python script that performs matrix decomposition, emissions estimation, and demand scaling using both national and California-specific data.

## Overview

This project integrates:

- Official EPA models: `useeior`, `stateior`, `LCIAformatter`
- CARB-specific flow modifications using `flowsa_CARB_version`
- A custom Python script (`run_model.py`) for computing California-specific emissions and demand matrices at the detailed NAICS level

The model estimates total demand and greenhouse gas (GHG) emissions for:

- The United States (detailed level)
- California (summary and detailed levels)
- Rest of the U.S. (RoUS)

## Repository Structure

```
USEEIO_CARB_Runner/
├── run_model.py                    # Main execution script
├── USEEIO.py                       # Python interface to useeior
├── environment.yml                 # Conda environment file
├── build_all_stateio_years.R       # R script to generate stateior output data
└── modelspecs/
    ├── bea_model_us_detailed_2017.yml
    └── bea_model_ca_summary_2022.yml
```

## Prerequisites

- This repository assumes you have [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed.
- R must be installed separately from [CRAN](https://cran.r-project.org/).
- Microsoft Visual C++ Redistributable for Visual Studio 2015–2022 may be required for some R and Python packages to compile successfully: [Download here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Quick Start (Windows Only)

Once you've installed Conda and cloned this repository, you can use run_setup_and_model.bat to:

- Activate the environment
- Install FLOWSA
- Set required R environment variables
- Launch the model script

You'll still need to manually install useeior and stateior in R. 

For a step-by-step setup, use the following instructions.


## Installation

1. **Clone this repository**

```bash
git clone https://github.com/leoal2/USEEIO_CARB_Runner.git
cd USEEIO_CARB_Runner
```

2. **Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate buildings
```

3. **Install R (separately) and required R packages**

We recommend installing R separately from CRAN instead of relying on Conda’s R, due to compatibility issues with some packages.
Download and install R from [https://cran.r-project.org](https://cran.r-project.org).

Then open R and run:

```r
install.packages("devtools", type = "win.binary")
devtools::install_github("USEPA/useeior")
devtools::install_github("USEPA/stateior")
```

If installation fails due to missing packages like `miniUI`, `pkgload`, or `shiny`, install them individually using:

```r
install.packages("shiny", type = "win.binary")
install.packages("pkgload", type = "win.binary")
install.packages("htmlwidgets", type = "win.binary")
# ...and so on
```

4. **Set R environment variables (Windows users only)**

Set the following variables manually or add them to your terminal configuration:

```bash
set R_HOME=%ProgramFiles%\R\R-<version>
set R_USER=%UserProfile%\Documents
set R_LIBS_USER=%LocalAppData%\Programs\R\R-<version>\library
```

These ensure that Python and `rpy2` correctly locate your R installation.

## Manual Setup Requirements

### 1. Copy YAML model spec files

Copy your model spec files into the following R folder:

```
copy modelspecs\*.* %R_HOME%\library\useeior\extdata\modelspecs
```

Example of required files:

- `bea_model_us_detailed_2017.yml`
- `bea_model_ca_summary_2022.yml`

### 2. Generate CA-specific emissions files using FLOWSA

Run the following in Python:

```python
import flowsa
from flowsa.flowbyactivity import getFlowByActivity
df_fba_ca = getFlowByActivity("StateGHGI_CA", 2022)

from flowsa.flowbysector import getFlowBySector
df_fbs_m1 = getFlowBySector("GHG_state_2022_m1")
df_fbs_ca = getFlowBySector("GHGc_state_CA_2022")
```

Generates:

```
flowsa/FlowBySector/GHGc_state_CA_2022_<version>.parquet
```

This file is required by the model YAML specifications in `useeior`.

### 3. Ensure LCIAformatter `.parquet` files exist

Make sure these files are present:

```
lciafmt/ipcc/IPCC_<version>.parquet
lciafmt/traci/TRACI_<version>.parquet
```

These are referenced in the `Indicators:` section of the YAML model files.

### 4. (Optional) Generate `stateior` outputs locally if S3 access fails

Pre-generated `.rds` files are included in the repository under `stateio_data/`.
When you run the Python scripts, they will automatically check if the required `.rds` files are present in your `%LocalAppData%\stateio` directory.
If any are missing, the scripts will copy them over automatically.

If automatic downloading of `.rds` files fails, you can generate them yourself by running the provided R script:

```r
source(paste0(Sys.getenv("USERPROFILE"), "/USEEIO_CARB_Runner/build_all_stateio_years.R"))
```

This will create the necessary `State_Summary_...` and `TwoRegion_Summary_...` `.rds` files in:

```
output_dir <- paste0(Sys.getenv("USERPROFILE"), "/AppData/Local/stateio")
```

## Running the Model

Once everything is set up, run:

```bash
python run_model.py
```

The script will:

- Load both US and California EEIO models
- Generate `L`, `A`, `D`, and `N` matrices
- Scale and disaggregate California demand
- Estimate GHG emissions for 2022 and adjust to 2017 USD
- Export results to Excel

## Outputs

The output will include:

- `CA_2022_2022USD_...xlsx`: results in 2022 dollars
- `CA_2022_2017USD_...xlsx`: results in CPI-adjusted 2017 dollars

Each file contains:

- Lifecycle matrices (L, A, D, N)
- Final demand vectors
- Sector-specific emissions for US, California, and RoUS

## Notes

- EPA dependencies like `esupy`, `stewi`, and `fedelemflowlist` are installed automatically via the `flowsa_CARB_version` fork.
- All `.yml` model specs must be placed in the correct `useeior` folder.
- `.parquet` indicator and satellite files must be available locally.
- You may need Microsoft Visual C++ Redistributables for Python/R interop via `rpy2`.

## Contact

This project is maintained by staff at the California Air Resources Board (CARB).

For questions, please contact [leoal2 on GitHub](https://github.com/leoal2).

