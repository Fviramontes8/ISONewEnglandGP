import numpy as np
import pandas as pd

def read_xls(xls_filename, sheet):
    return pd.read_excel(xls_filename, sheet_name=sheet)

def main():
    year = 2016
    NE_data = read_xls(f"xls_data/{year}_smd_hourly.xls", "ISONE CA")
    demand_numpy = np.array(NE_data["DEMAND"])
    #print(len(demand_numpy))
    np.save(f"ISONE_CA_DEMAND_{year}.npy", demand_numpy)

if __name__ == "__main__":
    main()
