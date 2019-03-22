import pandas as pd


# I: Import Data
us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_"
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_"
us_data_years = [1979, 1989, 1994, 1999, 2004, 2009, 2019]
intl_data_years = []

# US MERGER DATA
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

# INTERNATIONAL MERGER DATA
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

print("---- Data Sample ----")
print(US_MERGER_DATA.loc[1, :])
print("---- Shape ----")
print(US_MERGER_DATA.shape)
print("---- Col Names ----")
print(US_MERGER_DATA.columns)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())


# II: Remove "unnecessary" columns
US_MERGER_DATA = US_MERGER_DATA.drop(
    ["  Date Withdrawn", "  Date Effective", "Date Effective/ Unconditional",
     "Target Company  Date   of  Fin.", "History File Event"], axis=1
)

print("---- Shape ----")
print(US_MERGER_DATA.shape)
# print("---- Col Names ----")
# print(US_MERGER_DATA.columns)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())

US_MERGER_DATA2 = US_MERGER_DATA.dropna(axis=0)
print(US_MERGER_DATA2.shape)
print(US_MERGER_DATA2.loc[8141, :])