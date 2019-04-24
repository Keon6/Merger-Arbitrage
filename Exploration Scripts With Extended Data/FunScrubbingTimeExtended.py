import pandas as pd
import re
import numpy as np

# I: Import Data
# US MERGER DATA

us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_"
us_data_years = list(range(2001, 2021, 2))

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
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_"
intl_data_years = [1979, 1989]
intl_data_years.extend(list(range(1989, 2020, 2)))

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


# II: Data Scrubbing
# -> Drop points with missing labels
US_MERGER_DATA = US_MERGER_DATA.drop(US_MERGER_DATA[US_MERGER_DATA['Status'].isnull()].index.tolist())

# -> Explore labels
status = set()
for i in US_MERGER_DATA.index:
    status.add(US_MERGER_DATA.at[i, "Status"])

#  {'Withdrawn', 'S Buyer', 'Part Comp', 'Status Unknown', 'Pending Reg.', 'Intent W', 'Intended', 'Pending', 'Completed', 'S Target'}
# Success: Completed
# Failed: Withdrawn
# Not Useful: Part Comp, Pending, Pending Reg. Status Unknown, Intended, S Buyer, S Target, Intent W
print("---- Label Counts ----")
print("Withdrawl", (US_MERGER_DATA["Status"] == "Withdrawn").sum())
print("Part Comp", (US_MERGER_DATA.Status == "Part Comp").sum())
print("Completed", (US_MERGER_DATA.Status == "Completed").sum())
# ---- Label Counts ----
# Withdrawl 7699
# Part Comp 3
# Completed 114263

# -> Drop points with "un-useful" labels:
status.remove("Withdrawn")
status.remove("Completed")
remove_indices = []
for stat in status:
    remove_indices.extend(US_MERGER_DATA.index[US_MERGER_DATA['Status'] == stat].tolist())
US_MERGER_DATA = US_MERGER_DATA.drop(remove_indices)


# -> Regulatory Agency column needs to be separated and 1-hot encoded
regulatory_agencies = set()
for i in US_MERGER_DATA.index:
    s = str(US_MERGER_DATA.at[i, "Regulatory  Agencies"])
    if s.find("\n") != -1:
        # s.split("\n")
        for ra in s.split("\n"):
            regulatory_agencies.add(ra)
    else:
        regulatory_agencies.add(s)

for ra in regulatory_agencies:
    US_MERGER_DATA["Regulatory Agency_"+ra] = 0

for i in US_MERGER_DATA.index:
    s = str(US_MERGER_DATA.at[i, "Regulatory  Agencies"])
    if s.find("\n") != -1:
        for ra in s.split("\n"):
            US_MERGER_DATA.at[i, "Regulatory Agency_" + ra] = 1
    else:
        US_MERGER_DATA.at[i, "Regulatory Agency_" + s] = 1
US_MERGER_DATA = US_MERGER_DATA.drop(["Regulatory  Agencies"], axis=1)

# -> Reformat Binary Variables (Y/N -> 0/1)
binary_cols = {" Target Bankrupt", " Creeping Acquisition", "Asset Lockup", "Greenmail", "Target Lockup", "Poison  Pill",
               "Stock Lockup", "White Squire", "Divestiture", "DIVISION", " Dutch Auction Tender  Offer",
               "Financial  Acquiror", " Source    of   Fund- Borrowing", "Foreign Provider of Funds",
               " Source    of  Funds- Preferred   Stock   Issue", "Financing via Staple Offering",
               "Acquiror  is an Investor  Group", "LBO", " Joint Venture", "Litigation", "Liquid-  ation",
               "Acquiror Includes   Mgmt", "Mandatory Offering", "Merger   of Equals", "Privatization",
               "Privately Negotiated Purchases", "Reverse Takeover", "Self- Tender", "Target is a Financial Firm",
               "Target is a Leveraged Buyout Firm", " Target   is a Limited Partner-   ship",
               "Significant   Family  Ownership  of Target", "Tender Offer", "Tender/ Merger"}
for col in US_MERGER_DATA.columns:
    if col.find("Y/N") != -1:
        binary_cols.add(col)
    # 8
    elif col.find("Flag") != -1:
        binary_cols.add(col)
    # 7
    elif col.find("Defense") != -1 and col.find("Regulatory") == -1:
        binary_cols.add(col)
    # 10
    elif col.find("Source   of") != -1:
        binary_cols.add(col)
    # 8

# print(US_MERGER_DATA[binary_cols].loc[10:15])
for i in US_MERGER_DATA.index:
    for col in binary_cols:
        # print(col, US_MERGER_DATA[col].loc[100])
        if US_MERGER_DATA.at[i, col] == "N" or US_MERGER_DATA.at[i, col] == "No" \
                or US_MERGER_DATA.at[i, col] == "n" or US_MERGER_DATA.at[i, col] == "no":
            US_MERGER_DATA.at[i, col] = 0
        else:
            US_MERGER_DATA.at[i, col] = 1

print(US_MERGER_DATA[binary_cols].loc[10])


# -> Convert data types for each column as appropriate
US_MERGER_DATA["  Date Announced"] = pd.to_datetime(US_MERGER_DATA["  Date Announced"])
US_MERGER_DATA["  Date Withdrawn"] = pd.to_datetime(US_MERGER_DATA["  Date Withdrawn"])
US_MERGER_DATA["  Date Effective"] = pd.to_datetime(US_MERGER_DATA["  Date Effective"])
colnames_dtype_map = {
    "Status": "category",
    "Type  of  Acquiror": "category",
    "Acquiror Macro Industry": "category",
    "Acquiror Mid Industry": "category",
    "Acquiror  Nation": "category",
    "Acquiror Nation of Primary Stock Exchange (Name)": "category",
    "Acquiror's price per share": "float64",
    "Acquiror Number of Employees": "float64",
    " Target Goodwill": "float64",
    "  Target Net Cash Fr.  Investing": "float64",
    "Target Macro Industry": "category",
    "Target Mid Industry": "category",
    "Target Nation of Primary Stock Exchange (Name)": "category",
    "Target Number of Employees": "float64",
    "Target Nation": "category",
    "Initial reccode": "category"
}
for col in US_MERGER_DATA.columns:
    if col.find("mil") != -1:
        colnames_dtype_map[col] = "float64"
    elif col.find("$") != -1:
        colnames_dtype_map[col] = "float64"
    elif col.find("%") != -1:
        colnames_dtype_map[col] = "float64"
for col, dtype in colnames_dtype_map.items():
    # replace all commas in numbers and 'np' and 'nm' values to null
    if dtype == "float64":
        for i in US_MERGER_DATA.index:
            if type(US_MERGER_DATA.at[i, col]) is float:
                pass
            elif type(US_MERGER_DATA.at[i, col]) is str:
                if re.search('^[\D]*$', US_MERGER_DATA.at[i, col]):
                    US_MERGER_DATA.at[i, col] = np.nan
                else:
                    US_MERGER_DATA.at[i, col] = US_MERGER_DATA.at[i, col].replace(",", "")
    US_MERGER_DATA[col] = US_MERGER_DATA[col].astype(dtype)


# -> 1-hot encoding for categorical variables
categorical_cols = ["Type  of  Acquiror", "Initial reccode",
                    "Acquiror Macro Industry", "Acquiror Mid Industry", "Acquiror  Nation",
                    "Acquiror Nation of Primary Stock Exchange (Name)",
                    "Target Macro Industry", "Target Mid Industry", "Target Nation",
                    "Target Nation of Primary Stock Exchange (Name)"]
new = pd.get_dummies(US_MERGER_DATA[categorical_cols], drop_first=False, sparse=True)
for new_col in new.columns:
    US_MERGER_DATA[new_col] = new[new_col]
US_MERGER_DATA = US_MERGER_DATA.drop(columns=categorical_cols)
del new


# -> Rename columns to make them more legible
US_MERGER_DATA.rename(columns={
    '  Date Announced': 'Announced Date',
    "  Date Withdrawn": "Withdrawl Date",
    "  Date Effective": "Effective Date",
    ' % of Shares Acq.': '% of Shares Acquired',
    '  % Owned After Trans- action': '% Owned After Transaction',
    '  % sought': '% Sought',
    'Value of Transaction ($mil)': 'Transaction Value',
    "Type  of  Acquiror": "Acquiror Type",
    "Initial reccode": "Initial Response",
    "Acquiror  Nation": "Acquiror Nation",
    "Acquiror Nation of Primary Stock Exchange (Name)": "Acquiror Stock Exchange Nation",
    "Target Nation of Primary Stock Exchange (Name)": "Target Stock Exchange Nation",
},  inplace=True)


#### Feature Engineering
# Deal Length
US_MERGER_DATA['Deal Length (days)'] = US_MERGER_DATA['Effective Date'] - US_MERGER_DATA['Announced Date']
index = US_MERGER_DATA[US_MERGER_DATA['Deal Length (days)'].isnull()].index.tolist()
US_MERGER_DATA['Deal Length (days)'].loc[index] = (US_MERGER_DATA['Withdrawl Date'] - US_MERGER_DATA['Announced Date']).loc[index]
# Convert to days -> integers
US_MERGER_DATA['Deal Length (days)'] = US_MERGER_DATA['Deal Length (days)'].dt.days
index = US_MERGER_DATA[US_MERGER_DATA['Deal Length (days)'] == 0].index.tolist()
US_MERGER_DATA['Deal Length (days)'].loc[index] = np.nan


print("---- Shape ----")
print(US_MERGER_DATA.shape)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())


# III. Save as CSV:
US_MERGER_DATA.drop(['Effective Date', 'Announced Date', 'Withdrawl Date'], axis=1).to_csv(us_data_path+"Scrubbed_No_DefaultDistance_Extended.csv", sep=",", index=False)
