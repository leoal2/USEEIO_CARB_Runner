This guide assumes you are using Anaconda Prompt (Windows). Please install Miniconda or Anaconda before following these steps.

# USEEIO_CARB_Runner

This repository provides a California-customized implementation of the U.S. EPA's Environmentally-Extended Input-Output (USEEIO) model. It includes a Python script that performs matrix decomposition, emissions estimation, and demand scaling using both national and California-specific data.

## Overview

This project integrates:

* Official EPA models: `useeior`, `stateior`, `LCIAformatter`
* CARB-specific flow modifications using `flowsa_CARB_version`
* A custom Python script (`run_model.py`) for computing California-specific emissions and demand matrices at the detailed NAICS level

The model estimates total demand and greenhouse gas (GHG) emissions for:

* The United States (detailed level)
* California (summary and detailed levels)
* Rest of the U.S. (RoUS)

## Repository Structure

```
USEEIO_CARB_Runner/
├── run_model.py                    # Main execution script
├── USEEIO.py                       # Python interface to useeior
├── environment.yml                 # Conda environment file
├── build_all_stateio_years.R      # R script to generate stateior output data
└── modelspecs/
    ├── bea_model_us_detailed_2017.yml
    └── bea_model_ca_summary_2022.yml
```

## Prerequisites (IMPORTANT ORDER)

Before starting anything, follow this order:

1. **Install R 4.4.3**
   Download directly from CRAN (do NOT install R via Conda):
   [https://cran.r-project.org/bin/windows/base/](https://cran.r-project.org/bin/windows/base/)

   R 4.4.3 is recommended for full compatibility. R 4.5.0 may also work if you install R packages using RStudio, but could cause issues in `stateior` or `useeior` due to changes in data frame structure.

2. **Install Rtools 4.4**  
   Rtools is required to compile R packages like `useeior` and `stateior` when using `devtools`.

   Download the version that matches R 4.4.3:  
   [https://cran.rstudio.com/bin/windows/Rtools/rtools44/rtools.html]

   After installation, restart RStudio to ensure the toolchain is detected.

3. **Install RStudio Desktop** (free IDE for R):
   [https://posit.co/download/rstudio-desktop/](https://posit.co/download/rstudio-desktop/)

   Then:

   * Launch RStudio
   * Go to `Tools` → `Global Options` → `Packages`
   * Uncheck the box: **"Use secure download method for HTTP"**
   * Click "Apply"

4. **Install USEEIO and STATEIOR packages using RStudio**

   In RStudio Console:

   ```r
   install.packages("devtools")
   devtools::install_github("USEPA/useeior")
   devtools::install_github("USEPA/stateior")
   ```

   To confirm successful installation, run:

   ```r
   require(useeior)
   require(stateior)
   ```

   Both should return TRUE. If not, revisit the installation steps above.

5. **Open Anaconda Prompt**

6. **Ensure `git` is available inside Conda**

If using the zipped version of `flowsa_CARB_version`, you do NOT need `git` installed.

If using GitHub to clone the repository (instead of ZIP), install:

   ```bash
   conda install git
   ```

7. **Clone the repository (or extract the provided ZIP archive)**

If using the zipped version, unzip the `USEEIO_CARB_Runner.zip` file into a folder, and navigate into it:

   ```bash
   cd path\to\unzipped\USEEIO_CARB_Runner
   ```
If using GitHub, type the following:

   ```bash
   git clone https://github.com/leoal2/USEEIO_CARB_Runner.git
   cd USEEIO_CARB_Runner
   ```
   
8. **Create the Conda environment**

   If you previously created the `buildings` environment, remove it first:

   ```bash
   conda remove -n buildings --all
   ```

   Then:

   ```bash
   conda env create -f environment.yml
   conda activate buildings
   ```

9. **(If needed) Set R environment variables (Windows users only)**

    Only if `rpy2` cannot find your R installation, manually set:
    
    ```bash
    set R_HOME=C:\Program Files\R\R-4.4.3
    set R_USER=%UserProfile%\Documents
    set R_LIBS_USER=C:\Program Files\R\R-4.4.3\library
    ```

---

## Manual Setup Requirements

### 1. Copy YAML model spec files

Copy your model spec files into the following R folder:

```bash
copy modelspecs\*.* %R_HOME%\library\useeior\extdata\modelspecs
```

Required files:

* `bea_model_us_detailed_2017.yml`
* `bea_model_ca_summary_2022.yml`

### 2. Generate CA-specific emissions files using FLOWSA

If using the ZIP version of `flowsa_CARB_version`, make sure it's installed with pip:

```bash
pip install path\to\flowsa_CARB_version-main.zip
```

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

---

## Running the Model

Once everything is set up, run:

```bash
python run_model.py
```

The script will:

* Load both US and California EEIO models
* Generate `L`, `A`, `D`, and `N` matrices
* Scale and disaggregate California demand
* Estimate GHG emissions for 2022 and adjust to 2017 USD
* Export results to Excel

---

## Outputs

The output will include:

* `CA_2022_2022USD_...xlsx`: results in 2022 dollars
* `CA_2022_2017USD_...xlsx`: results in CPI-adjusted 2017 dollars

Each file contains:

* Lifecycle matrices (L, A, D, N)
* Final demand vectors
* Sector-specific emissions for US, California, and RoUS

---

## Notes

* EPA dependencies like `esupy`, `stewi`, and `fedelemflowlist` are installed automatically via the `flowsa_CARB_version` fork.
* All `.yml` model specs must be placed in the correct `useeior` folder.
* `.parquet` indicator and satellite files must be available locally.
* You may need Microsoft Visual C++ Redistributables for Python/R interop via `rpy2`.

**Why this installation order?**
Installing R and Rtools before Conda ensures that the correct system paths (like `R_HOME`) are set and persist when Conda activates its environments. This avoids common issues with misaligned paths when using `rpy2`.

---

## Important Notes on Runtime & SSL Setup

Some steps, like running FLOWSA or manually generating `stateior` .rds files, can take a significant amount of time. Please be patient and do not assume the script has failed if it appears inactive.

If you encounter SSL/TLS certificate errors when running Python or pip, install:

```bash
pip install pip-system-certs
```

---

## Troubleshooting

### R\_HOME not found

If `rpy2` fails to locate your R installation, manually set:

```bash
set R_HOME=C:\Program Files\R\R-4.4.3
```

---

### Generating `stateior` outputs locally if S3 access fails

If the model cannot download `.rds` files automatically, you can generate them manually:

```r
source(paste0(Sys.getenv("USERPROFILE"), "/USEEIO_CARB_Runner/build_all_stateio_years.R"))
```

Creates:

```
%LocalAppData%\stateio\TwoRegion_Summary_... .rds
%LocalAppData%\stateio\State_Summary_... .rds
```

---

## Contact

This project is maintained by staff at the California Air Resources Board (CARB).

For questions, please contact embodiedcarbon@arb.ca.gov.
