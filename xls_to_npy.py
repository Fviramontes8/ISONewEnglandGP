import numpy as np
import pandas as pd

def read_xls(xls_filename, sheet):
    return pd.read_excel(xls_filename, sheet_name=sheet)

def main():
    NE_data = read_xls("xls_data/2011_smd_hourly.xls", "ISONE CA")
    demand_numpy = np.array(NE_data["DEMAND"])
    #print(len(demand_numpy))
    np.save("ISONE_CA_DEMAND.npy", demand_numpy)

if __name__ == "__main__":
    main()
