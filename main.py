import pandas as pd
import numpy as np

import PlotUtils as pu

def read_xls(xls_filename, sheet):
    return pd.read_excel(xls_filename, sheet_name=sheet)

# Global variables are bad, m'kay?
def main():
    NH_data = read_xls("csv_data/2011_smd_hourly.xls", "NH")
    feature_data = np.array(
        [
            NH_data["DEMAND"], 
            NH_data["DryBulb"], 
            NH_data["DewPnt"]
        ]
    )
    feature_description = [
        "Non-PTF Demand",
        "Dry bulb temperature for the weather station",
        "Dew point temperature for the weather station"
    ]
    feature_units = [
        "$/MWh",
        "Temperature (Fahrenheit)",
        "Temperature (Fahrenheit)"
    ]
    print(f"Feature shape: {feature_data.shape}")

    for f_data, f_descript, f_units in zip(
        feature_data, 
        feature_description, 
        feature_units
    ): 
        pu.general_plot(f_data, f_descript, "Time (Hours)", f_units)
    

if __name__ == "__main__":
    main()
