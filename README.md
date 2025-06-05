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

````

## Prerequisites (IMPORTANT ORDER)

Before starting anything, follow this order:

1. **Install R 4.4.3**  
   Download directly from CRAN (do NOT install R via Conda):  
    [https://cran.r-project.org/bin/windows/base/](https://cran.r-project.org/bin/windows/base/)

2. **Install Rtools**  
   You’ll need Rtools to compile some packages.  
    [https://cran.r-project.org/bin/windows/Rtools/](https://cran.r-project.org/bin/windows/Rtools/)

3. **Install required R packages (`useeior`, `stateior`)**  
   In R (after installing R + Rtools), run:

```r
   install.packages("devtools", type = "win.binary")
   devtools::install_github("USEPA/useeior")
   devtools::install_github("USEPA/stateior")
````

4. **Install Miniconda or Anaconda**
    [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

5. **Make sure `git` is available inside Conda**
After activating Conda, run:

```bash
   conda install git
```

---

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

3. **(If needed) Set R environment variables (Windows users only)**

If you installed R before Conda, the `R_HOME` path is often automatically available and may not need to be reset. However, if you encounter Python/R interop issues, you can manually set:

```bash
set R_HOME=%LocalAppData%\Programs\R\R-4.4.3
set R_USER=%UserProfile%\Documents
set R_LIBS_USER=%LocalAppData%\Programs\R\R-4.4.3\library
```

These ensure that Python and `rpy2` correctly locate your R installation.

---

## Manual Setup Requirements

### 1. Copy YAML model spec files

Copy your model spec files into the following R folder:

```bash
copy modelspecs\*.* %R_HOME%\library\useeior\extdata\modelspecs
```

Example of required files:

* `bea_model_us_detailed_2017.yml`
* `bea_model_ca_summary_2022.yml`

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


To check that they exist, copy and paste this into Python:

```python
import os

ipcc_path = os.path.expandvars(r"%LocalAppData%/lciafmt/ipcc/IPCC_2023_100.parquet")
traci_path = os.path.expandvars(r"%LocalAppData%/lciafmt/traci/TRACI_2.1.parquet")

print("IPCC present:", os.path.exists(ipcc_path))
print("TRACI present:", os.path.exists(traci_path))
```

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
Installing R + Rtools before Conda ensures that the correct system paths (like `R_HOME`) are set and persist when Conda activates its environments.
This avoids common issues with missing compilers or misaligned paths when using `rpy2` or other Python-R bridges.

---

## Important Notes on Runtime & SSL Setup

Some steps, like running FLOWSA or manually generating stateior .rds files, can take a significant amount of time (sometimes several minutes).
This is normal — please be patient and do not assume the script has errored just because it appears inactive.

If you encounter SSL/TLS certificate errors when running Python or pip, install the pip-system-certs package to make sure Python uses your system’s trusted certificates:

```bash
pip install pip-system-certs
```

---

## Contact

This project is maintained by staff at the California Air Resources Board (CARB).

For questions, please contact [leoal2 on GitHub](https://github.com/leoal2).
