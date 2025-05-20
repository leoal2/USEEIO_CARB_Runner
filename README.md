# USEEIO_CARB_Runner

This repository provides a California-customized implementation of the U.S. EPA's Environmentally-Extended Input-Output (USEEIO) model. It includes a Python script that performs matrix decomposition, emissions estimation, and demand scaling using both national and California-specific data.

---

## Overview

This project integrates:

- Official EPA models: `useeior`, `stateior`, `LCIAformatter`
- CARB-specific flow modifications using `flowsa_CARB_version`
- A custom Python script (`run_model.py`) for computing California-specific emissions and demand matrices at the detailed NAICS level

The model estimates total demand and greenhouse gas (GHG) emissions for:

- The United States (detailed level)
- California (summary and detailed levels)
- Rest of the U.S. (RoUS)

---

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

---

## Key Dependencies

| Package              | Source                                                                                      |
|----------------------|---------------------------------------------------------------------------------------------|
| `useeior` (R)         | https://github.com/USEPA/useeior                                                           |
| `stateior` (R)        | https://github.com/USEPA/stateior                                                          |
| `LCIAformatter` (Python) | https://github.com/USEPA/LCIAformatter                                               |
| `flowsa_CARB_version` | https://github.com/leoal2/flowsa_CARB_version                                              |
| `fedelemflowlist`, `esupy`, `stewi`, `rpy2` | Installed via `environment.yml` or pip                            |

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

3. **Install modified FLOWSA package**

```bash
pip install git+https://github.com/leoal2/flowsa_CARB_version.git
```

4. **Install required R packages**

```r
install.packages("devtools")
devtools::install_github("USEPA/useeior")
devtools::install_github("USEPA/stateior")
```

---

## Manual Setup Requirements

### 1. Copy YAML model spec files

Copy the following YAML files into your local R library folder:

```
C:/Users/<username>/AppData/Local/Programs/R/R-4.4.2/library/useeior/extdata/modelspecs/
```

Files required:

- `bea_model_us_detailed_2017.yml`
- `bea_model_ca_summary_2022.yml`

---

### 2. Generate CA-specific emissions files using FLOWSA

Run in Python:

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
flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet
```

This is required by the configuration file ingested by useeior.

---

### 3. Ensure LCIAformatter `.parquet` files exist

Make sure these files are locally available:

```
lciafmt/ipcc/IPCC_v1.1.1_27ba917.parquet
lciafmt/traci/TRACI_2.1_v1.0.0_5555779.parquet
```

They are referenced by the `Indicators:` section in the YAML models.

---

### 4. (Optional) Run `stateior` locally if S3 downloads fail

If your system cannot download `.rds` files from Data Commons (Amazon S3), you must generate them locally.

Run the following R script:

```r
source("C:/Users/<username>/Downloads/build_all_stateio_years.R")
```

This script will automatically generate all `State_Summary_...` and `TwoRegion_Summary_...` `.rds` files, stored in:

```
C:/Users/<username>/AppData/Local/stateio
```

---

## Run the Model

```bash
python run_model.py
```

This script will:

- Load both US and California models
- Generate detailed `L`, `D`, `A`, and `N` matrices
- Scale and disaggregate summary-level data for California
- Estimate GHG emissions in both 2022 and CPI-adjusted 2017 dollars
- Output Excel files with all matrices and results

---

## Outputs

You will get two Excel workbooks:

- `CA_2022_2022USD_...xlsx` – with 2022 dollar values
- `CA_2022_2017USD_...xlsx` – adjusted to 2017 USD using CPI

Each file contains:

- L, A, D, N matrices (detailed + summary)
- Final demand and consumption vectors
- Sector-specific GHG emissions (US, CA, RoUS)

---

## Notes

- All `.yml` model specs must be correctly copied into the `useeior` R package folder.
- All `.parquet` indicator and satellite files must be present locally.
- If you modify `esupy` (e.g., to disable SSL verification), edit `esupy/remote.py` and set `verify=False` in `make_url_request()`.

---

## Contact

This project is maintained by California Air Resources Board (CARB) staff.

For questions or collaboration, contact [leoal2 on GitHub](https://github.com/leoal2).
```

