# USEEIO\_CARB\_Runner

This repository provides a complete, California-customized implementation of the U.S. EPA's Environmentally-Extended Input-Output (USEEIO) model.
It contains a runnable Python script that performs custom matrix decomposition, emissions estimation, and demand scaling using both national and California-specific data.

---

## 🗌 Overview

This project integrates:

* 📦 **Official EPA models**: `stateior`, `useeior`, `LCIAformatter`
* 🟡 **CARB modifications**: custom flow data and YAML configuration using `flowsa_BEC`
* 🧮 A **custom Python script (`run_model.py`)** for computing California-specific emissions and demand matrices at the detailed NAICS level

The model estimates total demand and GHG emissions for:

* **United States (Detailed)**
* **California (Summary & Detailed)**
* **Rest of U.S. (RoUS)**

---

## 📂 What This Repository Contains

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

---

## 🔧 Dependencies & Model Sources

| Component       | Source                                                        | Role                                                      |
| --------------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| `useeior`       | [USEPA/useeior](https://github.com/USEPA/useeior)             | Official EPA EEIO model engine (R)                        |
| `stateior`      | [USEPA/stateior](https://github.com/USEPA/stateior)           | Builds state-level IO tables (R)                          |
| `flowsa_BEC`    | [leoal2/flowsa\_BEC](https://github.com/leoal2/flowsa_BEC)    | **CARB-modified** flowsa version for custom GHG inventory |
| `LCIAformatter` | [USEPA/LCIAformatter](https://github.com/USEPA/LCIAformatter) | Official LCIA method formatting tool                      |
| `rpy2`          | [rpy2](https://rpy2.github.io/)                               | Connects Python to R                                      |

---

## 🛠 Prerequisites: Build Supporting Data

Before running `run_model.py`, you must first generate the required environmental and economic data using the following tools:

### Using `flowsa_BEC` (CARB-modified)

```python
from flowsa.flowbyactivity import getFlowByActivity 
df_fba_ca = getFlowByActivity("StateGHGI_CA", 2022)

from flowsa.flowbysector import getFlowBySector 
df_fbs_m1 = getFlowBySector("GHG_state_2022_m1")
df_fbs_ca = getFlowBySector("GHGc_state_CA_2022")
```

This produces a custom satellite table such as:

```
flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet
```

### Using `stateior`

StateIO must be run using a configuration file such as:

```
C:/Users/USERNAME/AppData/Local/Programs/R/R-4.4.2/library/stateior/extdata/modelspecs/StateIOv1.3-pecan.yml
```

This YAML file contains settings like:

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

This generates Make, Use, and trade-adjusted IO tables by state for each year.

### Using `LCIAformatter`

No action is needed if you're using the standard EPA files like:

```
lciafmt/ipcc/IPCC_v1.1.1_27ba917.parquet
lciafmt/traci/TRACI_2.1_v1.0.0_5555779.parquet
```

---

## 🧾 Model YAML Inputs for `useeior`

The USEEIO model is run based on YAML files located in:

```
~/AppData/Local/Programs/R/R-4.4.2/library/useeior/extdata/modelspecs/
```

### Example 1 — California 2022 Summary:

`bea_model_ca_summary_2022_after_IPCC.yml`

* `IODataSource`: `stateior`
* `ModelRegionAcronyms`: `[US-CA, RoUS]`
* `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4.parquet`

### Example 2 — USEEIO v2.3 National Detailed:

`USEEIOv2.3-GHG.yml`

* `IODataSource`: `BEA`
* `ModelRegionAcronyms`: `[US]`
* `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHG_national_2019_m2_v2.0.3.parquet`

### Example 3 — CAEEIO Pecan-style config:

`CAEEIOv1.3-pecan-22.yml`

* `BaseIOLevel`: `Summary`
* `Alias`: `pecan`
* `SatelliteTable.StaticFile`: `flowsa/FlowBySector/GHGc_state_CA_2022_v2.0.4_test.parquet`

These YAMLs drive the configuration and execution of the EEIO model and must be customized correctly before running.

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/leoal2/USEEIO_CARB_Runner.git
cd USEEIO_CARB_Runner
```

### 2. Set up the Conda environment

```bash
conda env create -f environment.yml
conda activate buildings
```

### 3. Install required R packages

Open R and run:

```r
install.packages("devtools")
devtools::install_github("USEPA/useeior")
devtools::install_github("USEPA/stateior")
devtools::install_github("USEEPA/LCIAformatter")
devtools::install_github("leoal2/flowsa_BEC")
```

---

## 🚀 Running the Model

After all prerequisites are satisfied and data generated, run:

```bash
python run_model.py
```

This will:

* Build national and California EEIO models
* Load Make/Use, final demand, and matrix data (L, D, A, N, x)
* Apply CARB-modified satellite data
* Generate detailed CA matrices and emissions
* Export results as Excel workbooks

---

## 📊 Outputs

The script generates two Excel files:

* `CA_2022_2022USD_...xlsx` — emissions and demand in 2022 USD
* `CA_2022_2017USD_...xlsx` — CPI-adjusted outputs in 2017 USD

Each file contains:

* Input matrices (L, A)
* Direct emissions factors (D)
* Lifecycle emissions factors (N)
* Demand vectors
* Sectoral GHG emissions for CA, RoUS, and US

---

## 💡 Notes

* `flowsa_BEC` must be used to generate updated CA-specific GHG parquet files
* `stateior` should be run with the correct configuration YAML to create `state` IO tables
* YAML files in `modelspecs/` define what data and regions are loaded
* `USEEIO.py` handles model orchestration and YAML overrides

---

## 📬 Contact

This work is maintained by CARB staff.
For questions, open an issue or contact [leoal2 on GitHub](https://github.com/leoal2).
