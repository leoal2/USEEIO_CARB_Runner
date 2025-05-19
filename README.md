# USEEIO\_CARB\_Runner

This repository provides a California-customized implementation of the U.S. EPA's Environmentally-Extended Input-Output (USEEIO) model. It includes a Python script to perform matrix decomposition, emissions estimation, and demand scaling using both national and California-specific data.

## Overview

This project integrates:

* Official EPA models: `stateior`, `useeior`, `LCIAformatter`
* CARB-specific modifications: custom flow data and YAML configuration using `flowsa_BEC`
* A custom Python script (`run_model.py`) for computing California-specific emissions and demand matrices at the detailed NAICS level

The model estimates total demand and greenhouse gas (GHG) emissions for:

* The United States (detailed level)
* California (summary and detailed levels)
* Rest of the U.S. (RoUS)

## Repository Structure

```
USEEIO_CARB_Runner/
├── run_model.py              # Main execution script
├── USEEIO.py                 # Python interface to useeior/stateior/flowsa
├── environment.yml           # Conda environment file
├── .gitignore
└── modelspecs/               # Model configuration files for useeior
    ├── bea_model_us_detailed_2017.yml
    └── bea_model_ca_summary_2022_after_IPCC.yml
```

## Dependencies and Model Sources

| Component         | Source                                                                            | Description                                                  |
| ----------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `useeior`         | [USEPA/useeior](https://github.com/USEPA/useeior)                                 | Official EEIO model engine implemented in R                  |
| `stateior`        | [USEPA/stateior](https://github.com/USEPA/stateior)                               | Builds state-level IO tables using a separate YAML spec      |
| `flowsa_CARB_version` | [leoal2/flowsa_CARB_version](https://github.com/leoal2/flowsa_CARB_version) | Fork of EPA's FLOWSA with CARB-specific method files and inventory updates |
| `LCIAformatter`   | [USEPA/LCIAformatter](https://github.com/USEPA/LCIAformatter)                     | Provides LCIA methods used by the model                      |
| `fedelemflowlist` | [USEPA/fedelemflowlist](https://github.com/USEPA/fedelemflowlist)                 | Used for mapping flows to standardized names and identifiers |
| `esupy`           | [USEPA/esupy](https://github.com/USEPA/esupy)                                     | Provides shared utilities across EPA SMM tools               |
| `stewi`           | [USEPA/standardizedinventories](https://github.com/USEPA/standardizedinventories) | Supports environmental inventories used in flowsa            |
| `rpy2`            | [rpy2](https://rpy2.github.io/)                                                   | Interface between Python and R                               |

Note: Only `flowsa_BEC` has been customized. The other dependencies (`fedelemflowlist`, `esupy`, and `stewi`) use the original EPA packages **except** that the file `fedelemflowlist/flowmapping/GHGI.csv` has been modified locally to incorporate CARB-specific mappings.

## Prerequisites: Build Supporting Data

### Step 1: Generate Environmental Data with Flowsa\_BEC

Run the following Python commands:

```python
from flowsa.flowbyactivity import getFlowByActivity 
df_fba_ca = getFlowByActivity("StateGHGI_CA", 2022)

from flowsa.flowbysector import getFlowBySector 
df_fbs_m1 = getFlowBySector("GHG_state_2022_m1")
df_fbs_ca = getFlowBySector("GHGc_state_CA_2022")
```

This will generate:

```
flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet
```

This file is referenced by the California YAML model spec used in `useeior`.

### Step 2: Build State IO Tables with stateior

In R, install and run `stateior` using a configuration file such as:

```
C:/Users/USERNAME/AppData/Local/Programs/R/R-4.4.2/library/stateior/extdata/modelspecs/StateIOv1.3-pecan.yml
```

Example key parameters:

```
Model: "StateIOv1.3-pecan"
BaseIOSchema: 2017
BaseIOLevel: "Summary"
model_ver: "0.4.0"
IOYear: [2022]
GeoScale: ["State", "TwoRegion"]
IODataSource: "BEA"
DataProduct: ["Make", "Use", "ValueAdded", "CommodityOutput"]
```

This produces state-level Make and Use tables that are used by `useeior` when `IODataSource: stateior` is set in the model configuration.

### Step 3: Ensure LCIAformatter Files are Available

You must have the following `.parquet` files locally:

```
lciafmt/ipcc/IPCC_v1.1.1_27ba917.parquet
lciafmt/traci/TRACI_2.1_v1.0.0_5555779.parquet
```

These files are referenced by the `Indicators` section of the model YAML.

## USEEIO YAML Specifications

The main `useeior` model is controlled by YAML files in:

```
~/AppData/Local/Programs/R/R-4.4.2/library/useeior/extdata/modelspecs/
```

Examples:

* `bea_model_us_detailed_2017.yml`: National detailed-level model (IODataSource: BEA)
* `bea_model_ca_summary_2022_after_IPCC.yml`: California summary-level model using CARB-modified GHG data (IODataSource: stateior)
* `CAEEIOv1.3-pecan-22.yml`: Two-region model for California + RoUS using `stateior` outputs and custom flowsa emissions

These files are edited dynamically via `USEEIO.py`, and must reference valid IOYears, BaseIOSchema, and satellite/indicator files.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/leoal2/USEEIO_CARB_Runner.git
cd USEEIO_CARB_Runner
```

2. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate buildings
```

3. Install required R packages:

```r
install.packages("devtools")
devtools::install_github("USEPA/useeior")
devtools::install_github("USEPA/stateior")
devtools::install_github("USEEPA/LCIAformatter")
pip install git+https://github.com/leoal2/flowsa_CARB_version.git
```

## Running the Model

Once prerequisites are complete:

```bash
python run_model.py
```

This script will:

* Load US and CA EEIO models
* Integrate IO and satellite data
* Compute detailed emissions factors and final demand
* Output results as Excel files

## Outputs

Two Excel files will be generated:

* `CA_2022_2022USD_...xlsx`: emissions and IO results in 2022 USD
* `CA_2022_2017USD_...xlsx`: CPI-adjusted results in 2017 USD

Each file contains:

* L, A, D, N matrices
* Demand vectors and CPI-adjusted consumption
* Sector-level GHG emissions (US, CA, RoUS)

## Notes

* All `.parquet` satellite/indicator files must be generated or downloaded locally
* `USEEIO.py` dynamically configures and edits YAML specs
* `flowsa_BEC` is the only package fully forked; the others use the latest EPA releases except for minor local adjustments to `fedelemflowlist`

## Contact

This project is maintained by CARB staff.
For questions or contributions, please contact [leoal2 on GitHub](https://github.com/leoal2).
