import pandas as pd
import re
import numpy as np

# I: Import Data
# US MERGER DATA
us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_"
us_data_years = [1979, 1989, 1994, 1999, 2004, 2009, 2019]

US_MERGER_DATA = pd.read_csv(us_data_path + f"{us_data_years[0]}_{us_data_years[1]}.csv")
for i in range(1, len(us_data_years)-1):
    US_MERGER_DATA = pd.concat(
        [
            US_MERGER_DATA,
            pd.read_csv(us_data_path + f"{us_data_years[i]}_{us_data_years[i+1]}.csv")
        ]
        , ignore_index=True
    )
US_MERGER_DATA.columns = [name.replace("\n", " ") for name in US_MERGER_DATA.columns]
# US_MERGER_DATA.replace("np", np.nan)
# US_MERGER_DATA.replace("nm", np.nan)

# print("---- Data Sample ----")
# print(US_MERGER_DATA.loc[1, :])
# print("---- Shape ----")
# print(US_MERGER_DATA.shape)
# print("---- Col Names ----")
# print(US_MERGER_DATA.columns)
# print("---- Null Count ----")
# print(US_MERGER_DATA.isnull().sum())

# INTERNATIONAL MERGER DATA
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_"
intl_data_years = []

# INTL_MERGER_DATA = pd.read_csv(intl_data_path + f"{intl_data_years[0]}_{intl_data_years[1]}.csv")
# for i in range(1, len(intl_data_years)-1):
#     INTL_MERGER_DATA = pd.concat(
#         [
#             INTL_MERGER_DATA,
#             pd.read_csv(intl_data_path + f"{intl_data_years[i]}_{intl_data_years[i+1]}.csv")
#         ]
#         , ignore_index=True
#     )
# INTL_MERGER_DATA.columns = [name.replace("\n", " ") for name in INTL_MERGER_DATA.columns]


# II: Remove "unnecessary" columns
US_MERGER_DATA = US_MERGER_DATA.drop(
    ["  Date Withdrawn", "  Date Effective", "Date Effective/ Unconditional", "Target Company  Date   of  Fin.",
     "History File Event", "Target Name", "Acquiror Name"], axis=1
)

# print("---- Shape ----")
# print(US_MERGER_DATA.shape)
# # print("---- Col Names ----")
# # print(US_MERGER_DATA.columns)
# print("---- Null Count ----")
# print(US_MERGER_DATA.isnull().sum())

# Convert data types for each column as appropriate
colnames = US_MERGER_DATA.columns
US_MERGER_DATA[" Rank Date"] = pd.to_datetime(US_MERGER_DATA[" Rank Date"])
US_MERGER_DATA["  Date Announced"] = pd.to_datetime(US_MERGER_DATA["  Date Announced"])
colnames_dtype_map = {
    "Target Industry Sector": "category",
    "Acquiror Industry Sector": "category",
    "Status": "category",
    "Value of Transaction ($mil)": "float64",
    "Ranking Value inc. Net Debt of Target ($Mil)": "float64",
    "Enterprise   Value   ($mil)": "float64",
    "  Equity Value      ($mil)": "float64",
    "Price  Per Share": "float64",
    'Target  Net Sales  LTM ($mil)': "float64",
    'EBIT Last Twelve Months ($ Mil)': "float64",
    'Pre-tax Income Last Twelve Months ($ Mil)': "float64",
    'Net Income Last Twelve Months ($ Mil)': "float64",
    'Target  Net Assets ($mil)': "float64",
    'Target Total Assets ($mil)': "float64",
    'Target  EBITDA  LTM ($mil)': "float64",
    'Target Book Value Per Share LTM (US$)': "float64",
    'Target Common Equity ($mil)': "float64",
    'Ratio of Offer Price to EPS': "float64"
}

for col, dtype in colnames_dtype_map.items():
    # replace all commas in numbers and 'np' and 'nm' values to null
    if dtype == "float64":
        # US_MERGER_DATA.loc[US_MERGER_DATA[col] == "np"] = np.nan
        # US_MERGER_DATA.loc[US_MERGER_DATA[col] == "nm"] = np.nan
        for i in US_MERGER_DATA.index:
            if type(US_MERGER_DATA.at[i, col]) is float:
                pass
            elif type(US_MERGER_DATA.at[i, col]) is str:
                if re.search('^[\D]*$', US_MERGER_DATA.at[i, col]):
                    US_MERGER_DATA.at[i, col] = np.nan
                else:
                    US_MERGER_DATA.at[i, col] = US_MERGER_DATA.at[i, col].replace(",", "")
    US_MERGER_DATA[col] = US_MERGER_DATA[col].astype(dtype)

print("---- Column Names and types ----")
for col in colnames:
    print("'" + col+"'", US_MERGER_DATA[col].dtypes)
#
#
print("---- Shape ----")
print(US_MERGER_DATA.shape)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())


