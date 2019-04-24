import pandas as pd
import numpy as np


us_data_path = "C:/Users/kevin/Desktop/US_Merger_Data_Scrubbed2.csv"
US_MERGER_DATA = pd.read_csv(us_data_path)


# print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())

# Feature Engineering
# Drop unnecessary columns
# US_MERGER_DATA = US_MERGER_DATA.drop(['Offer Price / EPS', 'Rank Date', 'Date Effective / Unconditional'], axis=1)


# 1) Net Debt = Enterprise Value - Equity Value
US_MERGER_DATA['Target Net Debt'] = US_MERGER_DATA['Target Enterprise Value'] - US_MERGER_DATA['Target Equity Value']
# # 2) Total Liabilities = Total Asset - Net Asset
# US_MERGER_DATA['Target Net Asset'] = US_MERGER_DATA['Target Total Asset'] - US_MERGER_DATA['Target Net Asset']


# Dates
# print(US_MERGER_DATA[['Announced Date', 'Effective Date', 'Withdrawl Date']].head())
US_MERGER_DATA['Announced Date'] = pd.to_datetime(US_MERGER_DATA['Announced Date'])
US_MERGER_DATA['Effective Date'] = pd.to_datetime(US_MERGER_DATA['Effective Date'])
US_MERGER_DATA['Withdrawl Date'] = pd.to_datetime(US_MERGER_DATA['Withdrawl Date'])

# 7)
US_MERGER_DATA['Deal Length (days)'] = US_MERGER_DATA['Effective Date'] - US_MERGER_DATA['Announced Date']
index = US_MERGER_DATA[US_MERGER_DATA['Deal Length (days)'].isnull()].index.tolist()
US_MERGER_DATA['Deal Length (days)'].loc[index] = (US_MERGER_DATA['Withdrawl Date'] - US_MERGER_DATA['Announced Date']).loc[index]
US_MERGER_DATA['Deal Length (days)'] = US_MERGER_DATA['Deal Length (days)'].dt.days
index = US_MERGER_DATA[US_MERGER_DATA['Deal Length (days)'] == 0].index.tolist()
US_MERGER_DATA['Deal Length (days)'].loc[index] = np.nan


# ##### END FEATURE ENGINEERING

# print(non_categorical_column_names)
# print("---- Shape ----")
# print(US_MERGER_DATA.shape)
# print("---- Null Count ----")
# print(US_MERGER_DATA.isnull().sum())


from Modules.UsefulFunctions import multivariate_gaussian_bayesian_imputation, multivariate_gaussian_bayesian_estimation


X = US_MERGER_DATA.drop(['Announced Date', 'Effective Date', 'Withdrawl Date', "Status", 'Offer Price / EPS', 'Rank Date', 'Date Effective / Unconditional'], axis=1)
Y = US_MERGER_DATA["Status"]
del US_MERGER_DATA


print(X.columns[:26])
non_categorical_column_names = list(X.columns[:26])
non_categorical_column_names.append(X.columns[-1])

# Impute Missing Data
X[non_categorical_column_names] = X[non_categorical_column_names].apply(np.log)
for col in non_categorical_column_names:
    inds = X[X[col] == -float("inf")].index.tolist()
    X[col].loc[inds] = np.nan


mu, Sigma = multivariate_gaussian_bayesian_estimation(X=X[non_categorical_column_names].dropna())
# print(mu)
# print(Sigma)
# print(mu.shape)
# print(Sigma.shape)
X[non_categorical_column_names] = multivariate_gaussian_bayesian_imputation(X=X[non_categorical_column_names], mu=mu, sigma=Sigma)
# print(X[X==np.nan].index)
# print(X.columns[:26])

# Feature engineering
# Balance Sheet

X[non_categorical_column_names] = X[non_categorical_column_names].apply(np.exp)


# 1) Net Debt = Enterprise Value - Equity Value
X['Target Net Debt'] = X['Target Enterprise Value'] - X['Target Equity Value']

# # 2) Total Liabilities = Total Asset - Net Asset
X['Target Net Asset'] = X['Target Total Asset'] - X['Target Net Asset']

# Income Sheet
# Rev = Operating Expense + EBITDA
# Rev = OE + D&A + IE + Tax + Net Income
# 3) OE = Rev - EBITDA
X['Target  Net Sales (YTD)'] = X['Target  Net Sales (YTD)'] - X['Target EBITDA (YTD)']

# 4) D&A = EBITDA - EBIT
X['Target EBITDA (YTD)'] = X['Target EBITDA (YTD)'] - X['Target EBIT (YTD)']

# 5) IE = EBIT - Pre-Tax Income
X['Target EBIT (YTD)'] = X['Target EBIT (YTD)'] - X['Target Pre-Tax Income (YTD)']

# 6) Tax = Pretax-Income - Net Income
X['Target Pre-Tax Income (YTD)'] = X['Target Pre-Tax Income (YTD)'] - X['Target Net Income (YTD)']

X.rename(columns={
    'Target  Net Sales (YTD)': 'Target Operating Expense (YTD)',
    'Target EBITDA (YTD)': 'Target Depreciation & Amortization (YTD)',
    'Target EBIT (YTD)': 'Target Interest Expense (YTD)',
    'Target Pre-Tax Income (YTD)': 'Target Tax (YTD)',
    'Target Net Asset': 'Target Total Liabilities'
},  inplace=True)


X.to_csv("C:/Users/kevin/Desktop/US_Merger_Data_Imputed.csv")




# # Data Split for CV
# kf = KFold(n_splits=10)
# NB_scores = list()  # score list for Naive Bayes
# LGR_scores = list()  # score list for Logistic Regression
# # skf = StratifiedKFold(n_splits=10)
# for train_indices, test_indices in kf.split(X):
#     X_train, X_test = X.loc[train_indices], X.loc[test_indices]
#     X_pca_train, X_pca_test = X_pca[train_indices], X_pca[test_indices]
#     Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
#
#     # I. PCA + Gaussian Naive Bayes
#     ngb = GaussianNB()
#     ngb.fit(X_pca_train, Y_train)
#     NB_scores.append(roc_auc_score(y_true=Y_test, y_score=ngb.predict(X_pca_test)))
#
#     # II. Logistic Regression with Elastic Net Regularization + SGD
#     lgr = SGDClassifier(loss="log", penalty="elasticnet")
#     lgr.fit(X_train, Y_train)
#     LGR_scores.append(roc_auc_score(y_true=Y_test, y_score=lgr.predict(X_pca_test)))
#
#     # III.
