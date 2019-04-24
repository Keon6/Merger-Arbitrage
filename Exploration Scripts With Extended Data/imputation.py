import pandas as pd
from Modules.UsefulFunctions import multivariate_gaussian_bayesian_imputation, multivariate_gaussian_bayesian_estimation
from sklearn.linear_model import LogisticRegression




us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Scrubbed_No_DefaultDistance_Extended.csv"

US_MERGER_DATA = pd.read_csv(us_data_path)
del us_data_path

# drop columns with too many missing values
drop_cols = []
for col in US_MERGER_DATA.columns:
    if US_MERGER_DATA.shape[0] - US_MERGER_DATA[col].isnull().sum() < 5000:
        print(col, US_MERGER_DATA[col].isnull().sum() )
        drop_cols.append(col)
US_MERGER_DATA = US_MERGER_DATA.drop(drop_cols, axis=1)

check = ["Transaction Value", 'Deal Length (days)', "Acquiror's price per share",
                                "Acquiror Number of Employees", "  Target Net Cash Fr.  Investing",
                                "Target Number of Employees"]
non_categorical_column_names = []
for col in US_MERGER_DATA.columns:
    if col in check:
        non_categorical_column_names.append(col)
    if col.find("mil") != -1:
        non_categorical_column_names.append(col)
    elif col.find("$") != -1:
        non_categorical_column_names.append(col)
    elif col.find("%") != -1:
        non_categorical_column_names.append(col)


d = 5
l, r = divmod(len(non_categorical_column_names), d)

from random import shuffle
shuffle(non_categorical_column_names)
partitions = []
for i in range(d):
    if i < d-1:
        partitions.append(non_categorical_column_names[i*l:(i+1)*l])
    elif i == d-1:
        partitions.append(non_categorical_column_names[i*l:])


# Impute Missing Data
# US_MERGER_DATA[non_categorical_column_names] = US_MERGER_DATA[non_categorical_column_names].apply(np.log)
#
# for col in non_categorical_column_names:
#     inds = US_MERGER_DATA[US_MERGER_DATA[col] == -float("inf")].index.tolist()
#     US_MERGER_DATA[col].loc[inds] = 0

for p in partitions:
    if US_MERGER_DATA[p].dropna().shape[0] < 30:
        print(US_MERGER_DATA[p].dropna().shape)
        raise Exception("Not enough data points for Imputation")

for p in partitions:
    print(US_MERGER_DATA[p].dropna().shape)
    mu, Sigma = multivariate_gaussian_bayesian_estimation(X=US_MERGER_DATA[p].dropna())
    US_MERGER_DATA[p] = multivariate_gaussian_bayesian_imputation(X=US_MERGER_DATA[p], mu=mu, sigma=Sigma)

# Save data
US_MERGER_DATA.to_csv("C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_ContImputed.csv")

# Impute Missing Binary Variables
check = [" Target Bankrupt", " Creeping Acquisition", "Asset Lockup", "Greenmail", "Target Lockup", "Poison  Pill",
               "Stock Lockup", "White Squire", "Divestiture", "DIVISION", " Dutch Auction Tender  Offer",
               "Financial  Acquiror", " Source    of   Fund- Borrowing", "Foreign Provider of Funds",
               " Source    of  Funds- Preferred   Stock   Issue", "Financing via Staple Offering",
               "Acquiror  is an Investor  Group", "LBO", " Joint Venture", "Litigation", "Liquid-  ation",
               "Acquiror Includes   Mgmt", "Mandatory Offering", "Merger   of Equals", "Privatization",
               "Privately Negotiated Purchases", "Reverse Takeover", "Self- Tender", "Target is a Financial Firm",
               "Target is a Leveraged Buyout Firm", " Target   is a Limited Partner-   ship",
               "Significant   Family  Ownership  of Target", "Tender Offer", "Tender/ Merger"]
binary_missing_cols = []
for col in US_MERGER_DATA.columns:
    if US_MERGER_DATA[col].isnull().sum() != 0 :
        if col in check:
            binary_missing_cols.append(col)
        elif col.find("Y/N") != -1:
            binary_missing_cols.append(col)
        elif col.find("Flag") != -1:
            binary_missing_cols.append(col)
        elif col.find("Defense") != -1 and col.find("Regulatory") == -1:
            binary_missing_cols.append(col)
        elif col.find("Source   of") != -1:
            binary_missing_cols.append(col)
# logistic regression Imputation
for col in binary_missing_cols:
    non_missing_inds = US_MERGER_DATA[US_MERGER_DATA[col].notnull()].index.tolist()
    missing_inds = US_MERGER_DATA[US_MERGER_DATA[col].isnull()].index.tolist()
    US_MERGER_DATA[col].at[missing_inds] = LogisticRegression().fit(
        X=US_MERGER_DATA[non_categorical_column_names].iloc[non_missing_inds],
        y=US_MERGER_DATA[col].iloc[non_missing_inds]
    ).predict(
        X=US_MERGER_DATA[non_categorical_column_names].iloc[missing_inds]
    )

US_MERGER_DATA.to_csv("C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Imputed.csv")
