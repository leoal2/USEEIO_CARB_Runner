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
└── modelspecs/               # Model configuration files
    ├── bea_model_us_detailed_2017.yml
    └── bea_model_ca_summary_2022_after_IPCC.yml
```

## Dependencies and Model Sources

| Component       | Source                                                        | Description                                            |
| --------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| `useeior`       | [USEPA/useeior](https://github.com/USEPA/useeior)             | Official EEIO model engine implemented in R            |
| `stateior`      | [USEPA/stateior](https://github.com/USEPA/stateior)           | Builds state-level IO tables                           |
| `flowsa_BEC`    | [leoal2/flowsa\_BEC](https://github.com/leoal2/flowsa_BEC)    | Modified version of flowsa including CA-specific flows |
| `LCIAformatter` | [USEPA/LCIAformatter](https://github.com/USEPA/LCIAformatter) | Provides LCIA methods used by the model                |
| `rpy2`          | [rpy2](https://rpy2.github.io/)                               | Interface between Python and R                         |

## Prerequisites: Build Supporting Data

### Flowsa\_BEC

Run the following Python commands to generate CA-specific flow and sector tables:

```python
from flowsa.flowbyactivity import getFlowByActivity 
df_fba_ca = getFlowByActivity("StateGHGI_CA", 2022)

from flowsa.flowbysector import getFlowBySector 
df_fbs_m1 = getFlowBySector("GHG_state_2022_m1")
df_fbs_ca = getFlowBySector("GHGc_state_CA_2022")
```

This will generate the following required file:

```
flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet
```

### StateIO

Run `stateior` using a configuration YAML such as:

```
C:/Users/USERNAME/AppData/Local/Programs/R/R-4.4.2/library/stateior/extdata/modelspecs/StateIOv1.3-pecan.yml
```

Key parameters of this YAML include:

```
Model: "StateIOv1.3-pecan"
BaseIOSchema: 2017
BaseIOLevel: "Summary"
model_ver: "0.4.0"
IOYear: [2012 ... 2023]
GeoScale: ["State", "TwoRegion"]
IODataSource: "BEA"
BasePriceType: "PRO"
DataProduct: ["Make", "Use", "DomesticUse", "ValueAdded", ...]
```

### LCIAformatter

No changes are required if you use the standard files:

```
lciafmt/ipcc/IPCC_v1.1.1_27ba917.parquet
lciafmt/traci/TRACI_2.1_v1.0.0_5555779.parquet
```

## USEEIO YAML Specifications

These are YAML files used by `useeior` to build the model, typically located in:

```
~/AppData/Local/Programs/R/R-4.4.2/library/useeior/extdata/modelspecs/
```

Examples:

* **California 2022 Summary**: `bea_model_ca_summary_2022_after_IPCC.yml`

  * `IODataSource`: `stateior`
  * `ModelRegionAcronyms`: `[US-CA, RoUS]`
  * `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet`

* **National Detailed**: `USEEIOv2.3-GHG.yml`

  * `IODataSource`: `BEA`
  * `ModelRegionAcronyms`: `[US]`
  * `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHG_national_2019_m2_v2.0.3.parquet`

* **CAEEIO Pecan Configuration**: `CAEEIOv1.3-pecan-22.yml`

  * `Alias`: `pecan`
  * `BaseIOLevel`: `Summary`
  * `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4_test.parquet`

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
devtools::install_github("leoal2/flowsa_BEC")
```

## Running the Model

Once prerequisites are complete, execute:

```bash
python run_model.py
```

This script will:

* Load national and California EEIO models
* Integrate Make/Use and emissions matrices
* Apply CARB custom satellite data
* Output Excel files with detailed GHG results

## Outputs

Two Excel files are generated:

* `CA_2022_2022USD_...xlsx`: results in 2022 USD
* `CA_2022_2017USD_...xlsx`: results adjusted to 2017 USD

Contents include:

* L, A, D, N matrices
* Final demand vectors
* Emissions by sector for US, CA, and RoUS

## Additional Notes

* All satellite and indicator parquet files must exist locally or be generated prior to model execution
* Model region and structure are defined by YAML files under `modelspecs/`
* `USEEIO.py` handles model setup and dynamic YAML editing as required

## Contact

This project is maintained by CARB staff.
For questions or contributions, please contact [leoal2 on GitHub](https://github.com/leoal2).
