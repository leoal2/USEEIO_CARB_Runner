import os
os.environ["R_HOME"] = "C:/Users/<username>/AppData/Local/Programs/R/R-4.4.2"
os.environ["STATEIOR_DATADIR"] = "C:/Users/<username>/AppData/Local/stateio" 

import pandas as pd
import USEEIO as EIO
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

ro.r('Sys.setenv(STATEIOR_DATADIR = "C:/Users/lguillot/AppData/Local/stateio")')

# Load USEEIO in R
useeior = importr("useeior")

def load_model(name, bea_year, ghg_year, region="US", detailed=True):
    try:
        model = EIO.Results(
            name=name,
            bea_year=bea_year,
            ghg_year=ghg_year,
            region=None if region == "US" else region,
            detailed=detailed,
            preserve=True  
        )
        return model
    except Exception as e:
        raise

def process_matrix(matrix, name):
    df = pd.DataFrame(matrix)
    df.insert(0, 'NAICS_Code', df.index)
    return df
    

def calculate_inputs_matrix(l_df, x_df, name):
    print(f"\n {name} -- sum of L values: {np.sum(l_df.iloc[:, 1:].values)}")
    print(f"{name} -- sum of x values: {x_df['x'].sum()}")
    l_columns = l_df.columns[1:]
    x_codes = x_df["NAICS_Code"]
    missing_codes = set(l_columns) - set(x_codes)
    x_dict = dict(zip(x_df["NAICS_Code"], x_df.iloc[:, 1]))
    aligned_x_values = [x_dict.get(code, 0) for code in l_columns]
    l_values = l_df.iloc[:, 1:].values
    x_values = pd.Series(aligned_x_values).values

    inputs_values = l_values * x_values
    inputs_df = pd.DataFrame(inputs_values, columns=l_columns, index=l_df["NAICS_Code"])
    inputs_df.insert(0, 'NAICS_Code', l_df["NAICS_Code"])

    print(f" {name} -- sum of resulting inputs matrix: {np.sum(inputs_values)}")

    return inputs_df

def calculate_emissions_matrix(inputs_df, d_df, name):
    inputs_rows = inputs_df["NAICS_Code"].tolist()
    d_columns = d_df.columns[1:].tolist()
    if inputs_rows != d_columns:
        raise ValueError(f"NAICS codes mismatch in {name} Inputs and D matrix. They must match row-wise")

    inputs_values = inputs_df.iloc[:, 1:].values
    d_values = d_df.iloc[0, 1:].values.reshape(-1, 1)

    emissions_values = inputs_values * d_values
    emissions_df = pd.DataFrame(emissions_values, columns=inputs_df.columns[1:], index=inputs_df["NAICS_Code"])
    emissions_df.insert(0, 'NAICS_Code', inputs_df["NAICS_Code"])

    return emissions_df

def calculate_scaling_factors(inputs_df, crosswalk_df, name):
    detail_to_summary = dict(zip(crosswalk_df["Detail"], crosswalk_df["Summary"]))

    input_naics_rows = inputs_df["NAICS_Code"].str.replace("/US", "", regex=False).tolist()
    input_naics_cols = inputs_df.columns[1:].str.replace("/US", "", regex=False).tolist()

    row_summary_map = {code: detail_to_summary.get(code, None) for code in input_naics_rows}
    col_summary_map = {code: detail_to_summary.get(code, None) for code in input_naics_cols}

    inputs_values = inputs_df.iloc[:, 1:].values
    scaling_factors = np.zeros_like(inputs_values, dtype=float)

    for i, row_naics in enumerate(input_naics_rows):
        for j, col_naics in enumerate(input_naics_cols):
            row_summary = row_summary_map[row_naics]
            col_summary = col_summary_map[col_naics]
            if row_summary and col_summary:
                k_indices = [idx for idx, naics in enumerate(input_naics_rows) if row_summary_map[naics] == row_summary]
                l_indices = [idx for idx, naics in enumerate(input_naics_cols) if col_summary_map[naics] == col_summary]
                total_inputs_sum = np.sum(inputs_values[np.ix_(k_indices, l_indices)])
                if total_inputs_sum != 0:
                    scaling_factors[i, j] = inputs_values[i, j] / total_inputs_sum
                else:
                    scaling_factors[i, j] = 0

    scaling_factors_df = pd.DataFrame(scaling_factors, columns=inputs_df.columns[1:], index=inputs_df["NAICS_Code"])
    scaling_factors_df.insert(0, 'NAICS_Code', inputs_df["NAICS_Code"])

    if scaling_factors_df.empty:
        print(" ERROR: Scaling factors returned an EMPTY DataFrame!")

    return scaling_factors_df

def calculate_ca_detailed_x(us_x_df, ca_summary_x_df, crosswalk_df, name="CA Detailed x"):
    us_x_df["NAICS_NoSuffix"] = us_x_df["NAICS_Code"].str.replace(r"/US$", "", regex=True)
    detail_to_summary = dict(zip(crosswalk_df["Detail"], crosswalk_df["Summary"]))
    summary_x_totals = {}

    for _, row in us_x_df.iterrows():
        detail_naics = row["NAICS_NoSuffix"]
        summary_naics = detail_to_summary.get(detail_naics, None)
        if summary_naics:
            summary_x_totals.setdefault(summary_naics, 0)
            summary_x_totals[summary_naics] += row["x"]
    scaling_factors = {}
    for _, row in us_x_df.iterrows():
        detail_naics = row["NAICS_NoSuffix"]
        summary_naics = detail_to_summary.get(detail_naics, None)
        if summary_naics and summary_naics in summary_x_totals and summary_x_totals[summary_naics] > 0:
            scaling_factors[detail_naics] = row["x"] / summary_x_totals[summary_naics]
        else:
            scaling_factors[detail_naics] = 0

    ca_detailed_x_values = []

    for _, row in us_x_df.iterrows():
        detail_naics = row["NAICS_NoSuffix"]
        summary_naics = detail_to_summary.get(detail_naics, None)
        if summary_naics:
            ca_summary_x_us_ca = ca_summary_x_df.loc[
                ca_summary_x_df["NAICS_Code"] == f"{summary_naics}/US-CA", "x"
            ].values[0] if f"{summary_naics}/US-CA" in ca_summary_x_df["NAICS_Code"].values else 0
            ca_detailed_x_us_ca = scaling_factors[detail_naics] * ca_summary_x_us_ca
            ca_detailed_x_values.append((detail_naics + "/US-CA", ca_detailed_x_us_ca))

    for _, row in us_x_df.iterrows():
        detail_naics = row["NAICS_NoSuffix"]
        summary_naics = detail_to_summary.get(detail_naics, None)
        if summary_naics:
            ca_summary_x_rous = ca_summary_x_df.loc[
                ca_summary_x_df["NAICS_Code"] == f"{summary_naics}/RoUS", "x"
            ].values[0] if f"{summary_naics}/RoUS" in ca_summary_x_df["NAICS_Code"].values else 0
            ca_detailed_x_rous = scaling_factors[detail_naics] * ca_summary_x_rous
            ca_detailed_x_values.append((detail_naics + "/RoUS", ca_detailed_x_rous))
    ca_detailed_x_df = pd.DataFrame(ca_detailed_x_values, columns=["NAICS_Code", "x"])

    return ca_detailed_x_df

def compute_inputs_detailed_ca(scaling_factors_df, inputs_summary_ca_df, crosswalk_df):
    scaling_factors_df.index = scaling_factors_df.index.str.replace(r'/US$', '', regex=True)
    scaling_factors_df.columns = scaling_factors_df.columns.str.replace(r'/US$', '', regex=True)

    inputs_summary_ca_df = inputs_summary_ca_df.loc[:, ~inputs_summary_ca_df.columns.duplicated()]
    scaling_factors_df = scaling_factors_df.loc[:, ~scaling_factors_df.columns.duplicated()]
    inputs_summary_ca_df = inputs_summary_ca_df[~inputs_summary_ca_df.index.duplicated(keep='first')]
    scaling_factors_df = scaling_factors_df[~scaling_factors_df.index.duplicated(keep='first')]

    detail_to_summary = dict(zip(crosswalk_df["Detail"], crosswalk_df["Summary"]))
    detailed_naics = scaling_factors_df.index.tolist()

    row_headers = [f"{code}/US-CA" for code in detailed_naics] + [f"{code}/RoUS" for code in detailed_naics]
    column_headers = [f"{code}/US-CA" for code in detailed_naics] + [f"{code}/RoUS" for code in detailed_naics]


    inputs_detailed_ca_df = pd.DataFrame(0.0, index=row_headers, columns=column_headers, dtype=float)

    for row_suffix in ["/US-CA", "/RoUS"]:
        for i in detailed_naics:
            for col_suffix in ["/US-CA", "/RoUS"]:
                for j in detailed_naics:
                    summary_i = detail_to_summary.get(i)
                    summary_j = detail_to_summary.get(j)
                    if summary_i is None or summary_j is None:
                        continue
                    scaling_factor = scaling_factors_df.loc[i, j] if (i in scaling_factors_df.index and j in scaling_factors_df.columns) else 0.0
                    input_summary_key_i = f"{summary_i}{row_suffix}"
                    input_summary_key_j = f"{summary_j}{col_suffix}"
                    input_summary_value = inputs_summary_ca_df.loc[input_summary_key_i, input_summary_key_j] if (
                        input_summary_key_i in inputs_summary_ca_df.index and input_summary_key_j in inputs_summary_ca_df.columns
                    ) else 0.0
                    inputs_detailed_ca_df.loc[f"{i}{row_suffix}", f"{j}{col_suffix}"] = float(scaling_factor * input_summary_value)

    inputs_detailed_ca_df.insert(0, "NAICS_Code", inputs_detailed_ca_df.index)
    return inputs_detailed_ca_df


def compute_detailed_l_ca(inputs_detailed_ca_df, ca_detailed_x_df, name="L_Detailed_CA"):
    x_dict = dict(zip(ca_detailed_x_df["NAICS_Code"], ca_detailed_x_df["x"]))
    missing_x_values = [j for j in inputs_detailed_ca_df.columns[1:] if j not in x_dict or x_dict[j] == 0]
    l_detailed_ca_df = inputs_detailed_ca_df.copy()

    for j in inputs_detailed_ca_df.columns[1:]:
        if j in x_dict and x_dict[j] != 0:
            l_detailed_ca_df[j] = inputs_detailed_ca_df[j] / x_dict[j]
        else:
            l_detailed_ca_df[j] = 0
    if "NAICS_Code" not in l_detailed_ca_df.columns:
        l_detailed_ca_df.insert(0, "NAICS_Code", inputs_detailed_ca_df["NAICS_Code"])
    return l_detailed_ca_df

def compute_detailed_d_ca(emissions_detailed_ca_df, inputs_detailed_ca_df, name="D_Detailed_CA"):
    sum_emissions = emissions_detailed_ca_df.iloc[:, 1:].sum(axis=1)
    sum_inputs = inputs_detailed_ca_df.iloc[:, 1:].sum(axis=1)
    sum_inputs.replace(0, np.nan, inplace=True)
    d_detailed_ca_values = sum_emissions / sum_inputs
    d_detailed_ca_df = pd.DataFrame({
        "NAICS_Code": emissions_detailed_ca_df["NAICS_Code"],
        "GHG": d_detailed_ca_values
    })
    d_detailed_ca_df["GHG"].fillna(0, inplace=True)
    return d_detailed_ca_df

def save_to_excel(sheets, output_file):
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def safe_insert_naics(df, reference_df):
    """ Inserts NAICS_Code into df if it doesn't already exist. """
    if "NAICS_Code" not in df.columns:
        df.insert(0, "NAICS_Code", reference_df["NAICS_Code"])
    return df



def main():
    try:
        # Load US and CA Models
        us_model = load_model(
            name="bea_model_us_detailed_2017",
            bea_year=2017,
            ghg_year=2017,
            region="US",
            detailed=True
        )
        ca_model = load_model(
            name="bea_model_ca_summary_2022_after_IPCC",
            bea_year=2022,
            ghg_year=2022,
            region="CA",
            detailed=False
        )

                # Process initial matrices
        l_us_detailed_df = process_matrix(us_model.L, "US Detailed L")
        x_us_detailed_df = process_matrix(us_model.x, "US Detailed x")
        d_d_us_detailed_df = process_matrix(us_model.D, "US Detailed D")

        l_ca_summary_df = process_matrix(ca_model.L, "CA Summary L")
        x_ca_summary_df = process_matrix(ca_model.x, "CA Summary x")
        d_d_ca_summary_df = process_matrix(ca_model.D, "CA Summary D")

        # Process A matrices
        a_us_detailed_df = process_matrix(us_model.A, "US Detailed A")
        a_ca_summary_df = process_matrix(ca_model.A, "CA Summary A")

        n_us_detailed_df = process_matrix(us_model.N, "US Detailed N")
        n_ca_summary_df = process_matrix(ca_model.N, "CA Summary N")

        # === Clean inputs from process_matrix (remove NAICS_Code column) ===
        n_df = n_us_detailed_df.drop(columns="NAICS_Code").set_index(us_model.N.index)
        l_df = l_us_detailed_df.drop(columns="NAICS_Code")
        l_df.columns = us_model.L.columns
        l_df.index = us_model.L.index
        L_inv = np.linalg.pinv(l_df.values)
        D_bc = n_df.values @ L_inv
        d_us_detailed_df = pd.DataFrame(D_bc, columns=l_df.columns)
        d_us_detailed_df.insert(0, "NAICS_Code", "Greenhouse Gases")

        # === CA Summary: Back-calculate D from N and L ===
        n_ca_df = n_ca_summary_df.drop(columns="NAICS_Code").set_index(ca_model.N.index)
        l_ca_df = l_ca_summary_df.drop(columns="NAICS_Code")
        l_ca_df.columns = ca_model.L.columns
        l_ca_df.index = ca_model.L.index
        L_ca_inv = np.linalg.pinv(l_ca_df.values)
        D_ca_bc = n_ca_df.values @ L_ca_inv
        d_ca_summary_df = pd.DataFrame(D_ca_bc, columns=l_ca_df.columns)
        d_ca_summary_df.insert(0, "NAICS_Code", "Greenhouse Gases")


        # Process crosswalk
        crosswalk_df = pd.DataFrame(us_model.crosswalk)
        crosswalk_df.insert(0, 'BEA_Code', crosswalk_df.index)

        # Compute Inputs, Emissions, and Scaling Factors
        print("\n Calling calculate_inputs_matrix with 2022 L and x...")
        inputs_us_detailed_df = calculate_inputs_matrix(l_us_detailed_df, x_us_detailed_df, "US Detailed")
        inputs_ca_summary_df = calculate_inputs_matrix(l_ca_summary_df, x_ca_summary_df, "CA Summary")

        emissions_us_detailed_df = calculate_emissions_matrix(inputs_us_detailed_df, d_us_detailed_df, "US Detailed")
        emissions_ca_summary_df = calculate_emissions_matrix(inputs_ca_summary_df, d_ca_summary_df, "CA Summary")

        a_inputs_ca_summary_df = calculate_inputs_matrix(a_ca_summary_df, x_ca_summary_df, "CA Summary 2 ")
        a_inputs_us_detailed_df = calculate_inputs_matrix(a_us_detailed_df, x_us_detailed_df, "US Detailed 2 ")
       
        n_emissions_us_detailed_df = calculate_emissions_matrix(a_inputs_us_detailed_df, n_us_detailed_df, "US Detailed N")
        n_emissions_ca_summary_df = calculate_emissions_matrix(a_inputs_ca_summary_df, n_ca_summary_df, "CA Summary N")

        scaling_factors_us_inputs_detailed_df = calculate_scaling_factors(inputs_us_detailed_df, crosswalk_df, "US Detailed")
        scaling_factors_us_emissions_detailed_df = calculate_scaling_factors(emissions_us_detailed_df, crosswalk_df, "US Detailed")

        a_scaling_factors_us_inputs_detailed_df = calculate_scaling_factors(a_inputs_us_detailed_df, crosswalk_df, "US Detailed 2")
        n_scaling_factors_us_emissions_detailed_df = calculate_scaling_factors(n_emissions_us_detailed_df, crosswalk_df, "US Detailed 2")

        ca_detailed_x_df = calculate_ca_detailed_x(x_us_detailed_df, x_ca_summary_df, crosswalk_df, "CA Detailed x")
        
        inputs_ca_detailed_df = compute_inputs_detailed_ca(scaling_factors_us_inputs_detailed_df, inputs_ca_summary_df, crosswalk_df)
        emissions_ca_detailed_df = compute_inputs_detailed_ca(scaling_factors_us_emissions_detailed_df, emissions_ca_summary_df, crosswalk_df)
        
        a_inputs_ca_detailed_df = compute_inputs_detailed_ca(a_scaling_factors_us_inputs_detailed_df, a_inputs_ca_summary_df, crosswalk_df)
        n_emissions_ca_detailed_df = compute_inputs_detailed_ca(n_scaling_factors_us_emissions_detailed_df, n_emissions_ca_summary_df, crosswalk_df)
        
        l_detailed_ca_df = compute_detailed_l_ca(inputs_ca_detailed_df, ca_detailed_x_df, "L_Detailed_CA")
        d_detailed_ca_df = compute_detailed_d_ca(emissions_ca_detailed_df, inputs_ca_detailed_df, "D_Detailed_CA")

        # Prepare L matrix and D vector (drop identifier columns)
        L_df = l_detailed_ca_df.set_index("NAICS_Code") if "NAICS_Code" in l_detailed_ca_df.columns else l_detailed_ca_df
        D_df = d_detailed_ca_df.set_index("NAICS_Code") if "NAICS_Code" in d_detailed_ca_df.columns else d_detailed_ca_df

        # Align D to L row index
        D_vector = D_df.loc[L_df.index, D_df.columns[0]]

        # Multiply each column in L by D_vector row-wise and sum
        N_series = L_df.multiply(D_vector, axis=0).sum(axis=0)

        # Convert to DataFrame
        n_detailed_ca_alt_df = pd.DataFrame({
            "NAICS_Code": N_series.index,
            "N_Detailed_CA_Alt": N_series.values
        }).set_index("NAICS_Code")

        
        a_detailed_ca_df = compute_detailed_l_ca(a_inputs_ca_detailed_df, ca_detailed_x_df, "A_Detailed_CA")
        n_detailed_ca_df = compute_detailed_d_ca(n_emissions_ca_detailed_df, a_inputs_ca_detailed_df, "N_Detailed_CA")


        # US Consumption
        us_consumption = us_model.consumption.get("Default")

        # CA Consumption (summary)
        ca_consumption = ca_model.consumption.get("CA")

        # Estimate CA Detailed Consumption
        crosswalk_clean = crosswalk_df[["Summary", "Detail"]].copy()
        us_detailed = us_consumption.copy()
        us_detailed.index = us_detailed.index.str.replace("/US", "", regex=False)
        us_detailed["Detail"] = us_detailed.index
        us_detailed = us_detailed.merge(crosswalk_clean, on="Detail", how="inner")

        summary_total = us_detailed.groupby("Summary")["TotalDemand"].transform("sum")
        us_detailed["ScalingFactor"] = us_detailed["TotalDemand"] / summary_total

        ca_summary = ca_consumption.copy()
        valid_index = ca_summary.index[ca_summary.index.str.contains("/")]
        split_index_df = pd.DataFrame(valid_index.str.rsplit("/", n=1, expand=True).to_list(),
                                    index=valid_index, columns=["Summary", "Region"])
        ca_summary["Summary"] = split_index_df["Summary"]
        ca_summary["Region"] = split_index_df["Region"]

        scaling_df = us_detailed[["Detail", "Summary", "ScalingFactor"]]
        ca_detailed = ca_summary.merge(scaling_df, on="Summary", how="inner")
        ca_detailed["EstimatedTotalDemand"] = ca_detailed["TotalDemand"] * ca_detailed["ScalingFactor"]
        ca_detailed.index = ca_detailed["Detail"] + "/" + ca_detailed["Region"]
        ca_detailed_final = ca_detailed[["EstimatedTotalDemand"]].rename(columns={"EstimatedTotalDemand": "TotalDemand"})

        common_codes_us = n_us_detailed_df.drop(columns="NAICS_Code").iloc[0].index.intersection(us_consumption["TotalDemand"].index)
        detailed_us_emissions_df = (n_us_detailed_df.drop(columns="NAICS_Code").iloc[0][common_codes_us] * us_consumption["TotalDemand"][common_codes_us]).reset_index()
        detailed_us_emissions_df.columns = ["NAICS_Code", "Detailed_US_Emissions"]

        common_codes = n_ca_summary_df.drop(columns="NAICS_Code").iloc[0].index.intersection(ca_consumption["TotalDemand"].index)
        summary_ca_emissions_df = (n_ca_summary_df.drop(columns="NAICS_Code").iloc[0][common_codes] * ca_consumption["TotalDemand"][common_codes]).reset_index()
        summary_ca_emissions_df.columns = ["NAICS_Code", "Summary_CA_Emissions"]

        # Reset both indexes
        ca_detailed_final = ca_detailed_final.reset_index()
        n_detailed_ca_alt_df = n_detailed_ca_alt_df.reset_index()

        # Merge using correct column names
        merged_df = pd.merge(
            ca_detailed_final, 
            n_detailed_ca_alt_df, 
            left_on="index", 
            right_on="NAICS_Code", 
            how="inner"
        )

        # Calculate emissions
        merged_df["Detailed_CA_Emissions"] = merged_df["TotalDemand"] * merged_df["N_Detailed_CA_Alt"]

        # Final output with consistent sector code column name
        detailed_ca_emissions_df = merged_df[["index", "Detailed_CA_Emissions"]].rename(columns={"index": "NAICS_Code"})

        # Save Initial Model Results
        sheets = {
            # Process initial matrices
            "US_Detailed_L": l_us_detailed_df,
            "US_Detailed_x": x_us_detailed_df,
            "US_Detailed_D": d_us_detailed_df,
            "US_Detailed_D_d": d_d_us_detailed_df,
            "CA_Summary_L": l_ca_summary_df,
            "CA_Summary_x": x_ca_summary_df,
            "CA_Summary_D": d_ca_summary_df,
            "CA_Summary_D_d": d_d_ca_summary_df,

            # Process A matrices
            "US_Detailed_A": a_us_detailed_df,
            "CA_Summary_A": a_ca_summary_df,
            "US_Detailed_N": n_us_detailed_df,
            "CA_Summary_N": n_ca_summary_df,
            
            # Process crosswalk
            "Crosswalk": crosswalk_df,
            
            # Compute Inputs and Emissions
            "Inputs_Detailed_US": inputs_us_detailed_df,
            "Inputs_Summary_CA": inputs_ca_summary_df,
            "Emissions_Detailed_US": emissions_us_detailed_df,
            "Emissions_Summary_CA": emissions_ca_summary_df,
            
            # Compute additional Inputs/Emissions for A and N matrices
            "A_Inputs_Detailed_US": a_inputs_us_detailed_df,
            "A_Inputs_Summary_CA": a_inputs_ca_summary_df,
            #"N_Emissions_Detailed_US": n_emissions_us_detailed_df,
            #"N_Emissions_Summary_CA": n_emissions_ca_summary_df,
            
            # Compute Scaling Factors
            "SF_Inputs_US": scaling_factors_us_inputs_detailed_df,
            "SF_Emissions_US": scaling_factors_us_emissions_detailed_df,
            "A_SF_Inputs_US": a_scaling_factors_us_inputs_detailed_df,
            #"N_SF_Emissions_US": n_scaling_factors_us_emissions_detailed_df,
            
            # Compute CA Detailed adjustments
            "CA_Detailed_x": ca_detailed_x_df,
            "Inputs_Detailed_CA": inputs_ca_detailed_df,
            "Emissions_Detailed_CA": emissions_ca_detailed_df,
            "A_Inputs_Detailed_CA": a_inputs_ca_detailed_df,
            #"N_Emissions_Detailed_CA": n_emissions_ca_detailed_df,
            
            # Compute final detailed matrices
            "L_Detailed_CA": l_detailed_ca_df,
            "D_Detailed_CA": d_detailed_ca_df,
            "A_Detailed_CA": a_detailed_ca_df,
            #"N_Detailed_CA": n_detailed_ca_df,
            "N_Detailed_CA_Alt" : n_detailed_ca_alt_df.reset_index(),

            "US_Consumption": us_consumption.reset_index(),
            "Detailed_US_Emissions": detailed_us_emissions_df,
            "US-CA_Summary_Consumption": ca_consumption.reset_index(),
            "Summary_CA_Emissions": summary_ca_emissions_df,
            "US-CA_Detailed_Consumption": ca_detailed_final.reset_index(),
            "Detailed_CA_Emissions": detailed_ca_emissions_df

        }

        output_excel_file = "CA_2022_2022USD.xlsx"
        save_to_excel(sheets, output_excel_file)

         # **Adjusting L and D Matrices to 2017 Dollars**
        print("\n Adjusting L and D matrices to 2017 dollars...")
        L_us_detailed_2017 = useeior.adjustResultMatrixPrice("L", 2017, False, us_model._model)
        D_us_detailed_2017 = useeior.adjustResultMatrixPrice("D", 2017, False, us_model._model)
        L_ca_summary_2017 = useeior.adjustResultMatrixPrice("L", 2017, False, ca_model._model)
        D_ca_summary_2017 = useeior.adjustResultMatrixPrice("D", 2022, False, ca_model._model)
        A_us_detailed_2017 = useeior.adjustResultMatrixPrice("A", 2017, False, us_model._model)
        N_us_detailed_2017 = useeior.adjustResultMatrixPrice("N", 2017, False, us_model._model)
        A_ca_summary_2017 = useeior.adjustResultMatrixPrice("A", 2017, False, ca_model._model)
        N_ca_summary_2017 = useeior.adjustResultMatrixPrice("N", 2017, False, ca_model._model)

        # Extract NAICS codes
        naics_codes_us = list(us_model._model.rx2["L"].rownames)
        naics_codes_ca = list(ca_model._model.rx2["L"].rownames)

        # Convert adjusted matrices to DataFrames
        l_us_detailed_2017_df = pd.DataFrame(np.array(L_us_detailed_2017), index=naics_codes_us, columns=naics_codes_us)
        x_us_detailed_2017_df = process_matrix(us_model.x, "US Detailed x")
        d_d_us_detailed_2017_df = pd.DataFrame(np.array(D_us_detailed_2017), index=["Greenhouse Gases"], columns=naics_codes_us)
        
        l_ca_summary_2017_df = pd.DataFrame(np.array(L_ca_summary_2017), index=naics_codes_ca, columns=naics_codes_ca)
        d_d_ca_summary_2017_df = pd.DataFrame(np.array(D_ca_summary_2017), index=["Greenhouse Gases"], columns=naics_codes_ca)

        print("\n CA Summary L matrix 2022 sample:")
        print(l_ca_summary_df.iloc[:5, :5])

        print("\n CA Summary L matrix 2017 sample:")
        print(l_ca_summary_2017_df.iloc[:5, :5])


        x_ca_summary_2017_df = pd.DataFrame(ca_model.x).rename(columns={0: "x"})
        x_ca_summary_2017_df.insert(0, "NAICS_Code", x_ca_summary_2017_df.index)

        # Get CPI matrix and calculate ratios
        cpi_matrix = ca_model._model.rx2("MultiYearCommodityCPI")
        cpi_array = np.array(cpi_matrix).T
        cpi_years = list(cpi_matrix.colnames)
        base_idx = cpi_years.index("2022")
        target_idx = cpi_years.index("2017")
        cpi_ratios = {
            sector: cpi_array[i, target_idx] / cpi_array[i, base_idx]
            for i, sector in enumerate(cpi_matrix.rownames)
        }

        # Apply CPI ratio to x
        x_ca_summary_2017_df["CPI"] = x_ca_summary_2017_df["NAICS_Code"].map(cpi_ratios).fillna(1.0)
        x_ca_summary_2017_df["x"] = x_ca_summary_2017_df["x"] * x_ca_summary_2017_df["CPI"]

        # Rename output variable
        x_ca_summary_2017_df = x_ca_summary_2017_df[["NAICS_Code", "x"]]

        print("\n First few rows of x_ca_summary_2017_df (should be adjusted):")
        print(x_ca_summary_2017_df.head())

        print("\n First few rows of x_ca_summary_df (original 2022):")
        print(x_ca_summary_df.head())

        print("\n Total sum of x values - 2022:", x_ca_summary_df["x"].sum())
        print(" Total sum of x values - 2017:", x_ca_summary_2017_df["x"].sum())



        a_us_detailed_2017_df = pd.DataFrame(np.array(A_us_detailed_2017), index=naics_codes_us, columns=naics_codes_us)
        a_ca_summary_2017_df = pd.DataFrame(np.array(A_ca_summary_2017), index=naics_codes_ca, columns=naics_codes_ca)
        
        n_us_detailed_2017_df = pd.DataFrame(np.array(N_us_detailed_2017), index=["Greenhouse Gases"], columns=naics_codes_us)
        n_ca_summary_2017_df = pd.DataFrame(np.array(N_ca_summary_2017), index=["Greenhouse Gases"], columns=naics_codes_ca)

        # Process crosswalk
        crosswalk_df = pd.DataFrame(us_model.crosswalk)
        crosswalk_df.insert(0, 'BEA_Code', crosswalk_df.index)

        # Reinsert NAICS codes
        for df in [l_us_detailed_2017_df, d_d_us_detailed_2017_df, l_ca_summary_2017_df, d_d_ca_summary_2017_df]:
            df.insert(0, "NAICS_Code", df.index)

        # Reinsert NAICS codes
        for df in [a_us_detailed_2017_df, n_us_detailed_2017_df, a_ca_summary_2017_df, n_ca_summary_2017_df]:
            df.insert(0, "NAICS_Code", df.index)

        # === Clean inputs from process_matrix (remove NAICS_Code column) ===
        n_df = n_us_detailed_2017_df.drop(columns="NAICS_Code").set_index(us_model.N.index)
        l_df = l_us_detailed_2017_df.drop(columns="NAICS_Code")
        l_df.columns = us_model.L.columns
        l_df.index = us_model.L.index
        L_inv = np.linalg.pinv(l_df.values)
        D_bc = n_df.values @ L_inv
        d_us_detailed_2017_df = pd.DataFrame(D_bc, columns=l_df.columns)
        d_us_detailed_2017_df.insert(0, "NAICS_Code", "Greenhouse Gases")

        # === CA Summary: Back-calculate D from N and L ===
        n_ca_df = n_ca_summary_2017_df.drop(columns="NAICS_Code").set_index(ca_model.N.index)
        l_ca_df = l_ca_summary_2017_df.drop(columns="NAICS_Code")
        l_ca_df.columns = ca_model.L.columns
        l_ca_df.index = ca_model.L.index
        L_ca_inv = np.linalg.pinv(l_ca_df.values)
        D_ca_bc = n_ca_df.values @ L_ca_inv
        d_ca_summary_2017_df = pd.DataFrame(D_ca_bc, columns=l_ca_df.columns)
        d_ca_summary_2017_df.insert(0, "NAICS_Code", "Greenhouse Gases")

        # Compute Adjusted Inputs and Emissions
        print("\n🧪 Calling calculate_inputs_matrix with 2017 L and x...")
        inputs_ca_summary_2017_df = calculate_inputs_matrix(l_ca_summary_2017_df, x_ca_summary_2017_df, "CA Summary (2017)")
        inputs_us_detailed_2017_df = calculate_inputs_matrix(l_us_detailed_2017_df, x_us_detailed_2017_df, "US Detailed (2017)")
        
        emissions_us_detailed_2017_df = calculate_emissions_matrix(inputs_us_detailed_2017_df, d_us_detailed_2017_df, "US Detailed (2017)")
        emissions_ca_summary_2017_df = calculate_emissions_matrix(inputs_ca_summary_2017_df, d_ca_summary_2017_df, "CA Summary (2017)")

        a_inputs_ca_summary_2017_df = calculate_inputs_matrix(a_ca_summary_2017_df, x_ca_summary_2017_df, "CA Summary 2 (2017)")

        a_inputs_us_detailed_2017_df = calculate_inputs_matrix(a_us_detailed_2017_df, x_us_detailed_2017_df, "US Detailed 2 (2017)")
        
        n_emissions_us_detailed_2017_df = calculate_emissions_matrix(a_inputs_us_detailed_2017_df, n_us_detailed_2017_df, "US Detailed N (2017)")
        n_emissions_ca_summary_2017_df = calculate_emissions_matrix(a_inputs_ca_summary_2017_df, n_ca_summary_2017_df, "CA Summary N (2017)")

        # Compute Scaling Factors and CA Detailed Adjustments
        scaling_factors_us_inputs_detailed_2017_df = calculate_scaling_factors(inputs_us_detailed_2017_df, crosswalk_df, "US Detailed (2017)")
        scaling_factors_us_emissions_detailed_2017_df = calculate_scaling_factors(emissions_us_detailed_2017_df, crosswalk_df, "US Detailed (2017)")

        a_scaling_factors_us_inputs_detailed_2017_df = calculate_scaling_factors(a_inputs_us_detailed_2017_df, crosswalk_df, "US Detailed 2 (2017)")
        n_scaling_factors_us_emissions_detailed_2017_df = calculate_scaling_factors(n_emissions_us_detailed_2017_df, crosswalk_df, "US Detailed 2 (2017)")

        ca_detailed_x_2017_df = calculate_ca_detailed_x(x_us_detailed_2017_df, x_ca_summary_2017_df, crosswalk_df, "CA Detailed x (2017)")

        inputs_ca_detailed_2017_df = compute_inputs_detailed_ca(scaling_factors_us_inputs_detailed_2017_df, inputs_ca_summary_2017_df, crosswalk_df)
        emissions_ca_detailed_2017_df = compute_inputs_detailed_ca(scaling_factors_us_emissions_detailed_2017_df, emissions_ca_summary_2017_df, crosswalk_df)

        a_inputs_ca_detailed_2017_df = compute_inputs_detailed_ca(a_scaling_factors_us_inputs_detailed_2017_df, a_inputs_ca_summary_2017_df, crosswalk_df)
        n_emissions_ca_detailed_2017_df = compute_inputs_detailed_ca(n_scaling_factors_us_emissions_detailed_2017_df, n_emissions_ca_summary_2017_df, crosswalk_df)

        l_detailed_ca_2017_df = compute_detailed_l_ca(inputs_ca_detailed_2017_df, ca_detailed_x_2017_df, "L_Detailed_CA (2017)")
        d_detailed_ca_2017_df = compute_detailed_d_ca(emissions_ca_detailed_2017_df, inputs_ca_detailed_2017_df, "D_Detailed_CA (2017)")

        # Prepare L (2017) matrix and D (2017) vector
        L_2017_df = l_detailed_ca_2017_df.set_index("NAICS_Code") if "NAICS_Code" in l_detailed_ca_2017_df.columns else l_detailed_ca_2017_df
        D_2017_df = d_detailed_ca_2017_df.set_index("NAICS_Code") if "NAICS_Code" in d_detailed_ca_2017_df.columns else d_detailed_ca_2017_df

        # Align D vector to L matrix index
        D_vector_2017 = D_2017_df.loc[L_2017_df.index, D_2017_df.columns[0]]

        # Multiply each column in L by D vector (element-wise) and sum across rows
        N_series_2017 = L_2017_df.multiply(D_vector_2017, axis=0).sum(axis=0)

        # Convert to a clean DataFrame
        n_detailed_ca_alt_2017_df = pd.DataFrame({
            "NAICS_Code": N_series_2017.index,
            "N_Detailed_CA_2017": N_series_2017.values
        }).set_index("NAICS_Code")


        a_detailed_ca_2017_df = compute_detailed_l_ca(a_inputs_ca_detailed_2017_df, ca_detailed_x_2017_df, "A_Detailed_CA  (2017)")
        n_detailed_ca_2017_df = compute_detailed_d_ca(n_emissions_ca_detailed_2017_df, a_inputs_ca_detailed_2017_df, "N_Detailed_CA (2017)")

        # === Extract CPI Ratios and Adjust Consumption ===
        print("\n📊 Extracting CPI ratios from 2022 to 2017...")

        us_cpi_matrix = us_model._model.rx2("MultiYearCommodityCPI")
        ca_cpi_matrix = ca_model._model.rx2("MultiYearCommodityCPI")

        us_cpi_array = np.array(us_cpi_matrix).T
        ca_cpi_array = np.array(ca_cpi_matrix).T

        us_cpi_years = list(us_cpi_matrix.colnames)
        ca_cpi_years = list(ca_cpi_matrix.colnames)
        us_sectors = list(us_cpi_matrix.rownames)
        ca_sectors = list(ca_cpi_matrix.rownames)

        base_year = "2022"
        target_year = "2017"

        us_base_idx = us_cpi_years.index(base_year)
        us_target_idx = us_cpi_years.index(target_year)

        ca_base_idx = ca_cpi_years.index(base_year)
        ca_target_idx = ca_cpi_years.index(target_year)

        # CPI ratios as full keys
        us_cpi_ratios = {
            f"{code}/US": us_cpi_array[i, us_target_idx] / us_cpi_array[i, us_base_idx]
            for i, code in enumerate(us_sectors)
        }

        ca_cpi_ratios = {
            code: ca_cpi_array[i, ca_target_idx] / ca_cpi_array[i, ca_base_idx]
            for i, code in enumerate(ca_sectors)
        }

        # === US Consumption (Adjusted to 2017 USD)
        us_consumption = us_model.consumption.get("Default").copy()
        us_consumption["CPI"] = us_consumption.index.to_series().apply(lambda code: us_cpi_ratios.get(code, 1.0))
        us_consumption["AdjustedDemand"] = us_consumption["TotalDemand"] * us_consumption["CPI"]
        us_consumption["Detail"] = us_consumption.index.str.replace("/US", "", regex=False)

        # === CA Summary Consumption (Adjusted to 2017 USD)
        ca_consumption = ca_model.consumption.get("CA").copy()
        ca_consumption["CPI"] = ca_consumption.index.to_series().apply(lambda code: ca_cpi_ratios.get(code, 1.0))
        ca_consumption["AdjustedDemand"] = ca_consumption["TotalDemand"] * ca_consumption["CPI"]
        ca_consumption["Summary"] = ca_consumption.index.str.split("/").str[0]
        ca_consumption["Region"] = ca_consumption.index.str.split("/").str[1]

        # === Estimate CA Detailed Adjusted Consumption
        crosswalk_clean = us_model.crosswalk[["Summary", "Detail"]].copy()
        us_detailed = us_consumption.copy()
        us_detailed = us_detailed.merge(crosswalk_clean, on="Detail", how="inner")

        summary_total = us_detailed.groupby("Summary")["AdjustedDemand"].transform("sum")
        us_detailed["ScalingFactor"] = us_detailed["AdjustedDemand"] / summary_total

        scaling_df = us_detailed[["Detail", "Summary", "ScalingFactor"]]
        ca_summary = ca_consumption.copy()

        ca_detailed = ca_summary.merge(scaling_df, on="Summary", how="inner")
        ca_detailed["EstimatedTotalDemand"] = ca_detailed["AdjustedDemand"] * ca_detailed["ScalingFactor"]
        ca_detailed.index = ca_detailed["Detail"] + "/" + ca_detailed["Region"]
        ca_detailed_final = ca_detailed[["EstimatedTotalDemand"]].rename(columns={"EstimatedTotalDemand": "TotalDemand"})

        common_codes_us_2017 = n_us_detailed_2017_df.drop(columns="NAICS_Code").iloc[0].index.intersection(us_consumption["AdjustedDemand"].index)
        detailed_us_emissions_2017_df = (n_us_detailed_2017_df.drop(columns="NAICS_Code").iloc[0][common_codes_us_2017] * us_consumption["AdjustedDemand"][common_codes_us_2017]).reset_index()
        detailed_us_emissions_2017_df.columns = ["NAICS_Code", "Detailed_US_Emissions"]

        common_codes_2017 = n_ca_summary_2017_df.drop(columns="NAICS_Code").iloc[0].index.intersection(ca_consumption["AdjustedDemand"].index)
        summary_ca_emissions_2017_df = (n_ca_summary_2017_df.drop(columns="NAICS_Code").iloc[0][common_codes_2017] * ca_consumption["AdjustedDemand"][common_codes_2017]).reset_index()
        summary_ca_emissions_2017_df.columns = ["NAICS_Code", "Summary_CA_Emissions"]

        # Reset both indexes
        ca_detailed_final = ca_detailed_final.reset_index()
        n_detailed_ca_alt_df = n_detailed_ca_alt_df.reset_index()

        # Merge using correct column names
        merged_df = pd.merge(
            ca_detailed_final, 
            n_detailed_ca_alt_2017_df, 
            left_on="index", 
            right_on="NAICS_Code", 
            how="inner"
        )

        # Calculate emissions
        merged_df["Detailed_CA_Emissions"] = merged_df["TotalDemand"] * merged_df["N_Detailed_CA_2017"]

        # Final output with consistent sector code column name
        detailed_ca_emissions_2017_df = merged_df[["index", "Detailed_CA_Emissions"]].rename(columns={"index": "NAICS_Code"})


        # Create a dictionary of sheets for the 2017 adjusted matrices
        sheets_adjusted = {
            # Adjusted initial matrices (L, x, D)
            "US_Detailed_L_2017": l_us_detailed_2017_df,
            "US_Detailed_x_2017": x_us_detailed_2017_df,
            "US_Detailed_D_2017": d_us_detailed_2017_df,
            "US_Detailed_D_d_2017": d_d_us_detailed_2017_df,
            "CA_Summary_L_2017": l_ca_summary_2017_df,
            "CA_Summary_x_2017": x_ca_summary_2017_df,
            "CA_Summary_D_2017": d_ca_summary_2017_df,
            "CA_Summary_D_d_2017": d_d_ca_summary_2017_df,
            
            # Adjusted A and N matrices
            "US_Detailed_A_2017": a_us_detailed_2017_df,
            "CA_Summary_A_2017": a_ca_summary_2017_df,
            "US_Detailed_N_2017": n_us_detailed_2017_df,
            "CA_Summary_N_2017": n_ca_summary_2017_df,
            
            # Include crosswalk (as processed for 2017)
            "Crosswalk_2017": crosswalk_df,
            
            # Compute Adjusted Inputs and Emissions
            "Inputs_Detailed_US_2017": inputs_us_detailed_2017_df,
            "Inputs_Summary_CA_2017": inputs_ca_summary_2017_df,
            "Emissions_Detailed_US_2017": emissions_us_detailed_2017_df,
            "Emissions_Summary_CA_2017": emissions_ca_summary_2017_df,
            
            # Compute additional Inputs/Emissions for adjusted A and N matrices
            "A_Inputs_Detailed_US_2017": a_inputs_us_detailed_2017_df,
            "A_Inputs_Summary_CA_2017": a_inputs_ca_summary_2017_df,
            #"N_Emissions_Detailed_US_2017": n_emissions_us_detailed_2017_df,
            #"N_Emissions_Summary_CA_2017": n_emissions_ca_summary_2017_df,
            
            # Compute Scaling Factors for adjusted matrices
            "SF_Inputs_US_2017": scaling_factors_us_inputs_detailed_2017_df,
            "SF_Emissions_US_2017": scaling_factors_us_emissions_detailed_2017_df,
            "A_SF_Inputs_US_2017": a_scaling_factors_us_inputs_detailed_2017_df,
            #"N_SF_Emissions_US_2017": n_scaling_factors_us_emissions_detailed_2017_df,
            
            # Compute CA Detailed adjustments for the 2017 matrices
            "CA_Detailed_x_2017": ca_detailed_x_2017_df,
            "Inputs_Detailed_CA_2017": inputs_ca_detailed_2017_df,
            "Emissions_Detailed_CA_2017": emissions_ca_detailed_2017_df,
            "A_Inputs_Detailed_CA_2017": a_inputs_ca_detailed_2017_df,
            #"N_Emissions_Detailed_CA_2017": n_emissions_ca_detailed_2017_df,
            
            # Final detailed CA matrices
            "L_Detailed_CA_2017": l_detailed_ca_2017_df,
            "D_Detailed_CA_2017": d_detailed_ca_2017_df,
            "A_Detailed_CA_2017": a_detailed_ca_2017_df,
            #"N_Detailed_CA_2017": n_detailed_ca_2017_df,
            "N_Detailed_CA_Alt_2017" : n_detailed_ca_alt_2017_df.reset_index(),

            "US_Consumption_2017": us_consumption.reset_index(),
            "Detailed_US_Emissions_2017": detailed_us_emissions_2017_df,
            "US-CA_Summary_Consumption_2017": ca_consumption.reset_index(),
            "Summary_CA_Emissions_2017": summary_ca_emissions_2017_df,
            "US-CA_Detailed_Consumption_2017": ca_detailed_final.reset_index(),
            "Detailed_CA_Emissions_2017": detailed_ca_emissions_2017_df

        }

        # Save all the adjusted sheets to the Excel file
        output_excel_file_adjusted = "CA_2022_2017USD.xlsx"
        save_to_excel(sheets_adjusted, output_excel_file_adjusted)

    
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
