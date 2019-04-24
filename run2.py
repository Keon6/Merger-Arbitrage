import pandas as pd
import numpy as np
from Modules.Models.LocalGP import LGPC_CV, LocalGaussianProcessClassifier
from sklearn.gaussian_process.kernels import (Matern, RationalQuadratic,
                                              ExpSineSquared, RBF, ConstantKernel,
                                              Product, Sum, WhiteKernel)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


us_data_path = "C:/Users/kevin/Desktop/US_Merger_Data_Scrubbed2.csv"

US_MERGER_DATA = pd.read_csv(us_data_path)
del us_data_path

# print("---- Null Count ----")
# print(US_MERGER_DATA.isnull().sum())

# Feature Engineering
# Drop unnecessary columns
# US_MERGER_DATA = US_MERGER_DATA.drop(['Offer Price / EPS', 'Rank Date', 'Date Effective / Unconditional'], axis=1)


# 1) Net Debt = Enterprise Value - Equity Value
US_MERGER_DATA['Target Net Debt'] = US_MERGER_DATA['Target Enterprise Value'] - US_MERGER_DATA['Target Equity Value']
# # 2) Total Liabilities = Total Asset - Net Asset
# US_MERGER_DATA['Target Net Asset'] = US_MERGER_DATA['Target Total Asset'] - US_MERGER_DATA['Target Net Asset']

# 1) Net Debt = Enterprise Value - Equity Value
US_MERGER_DATA['Target Net Debt'] = US_MERGER_DATA['Target Enterprise Value'] - US_MERGER_DATA['Target Equity Value']

# # 2) Total Liabilities = Total Asset - Net Asset
US_MERGER_DATA['Target Net Asset'] = US_MERGER_DATA['Target Total Asset'] - US_MERGER_DATA['Target Net Asset']

# Income Sheet
# Rev = Operating Expense + EBITDA
# Rev = OE + D&A + IE + Tax + Net Income
# 3) OE = Rev - EBITDA
US_MERGER_DATA['Target  Net Sales (YTD)'] = US_MERGER_DATA['Target  Net Sales (YTD)'] - US_MERGER_DATA['Target EBITDA (YTD)']

# 4) D&A = EBITDA - EBIT
US_MERGER_DATA['Target EBITDA (YTD)'] = US_MERGER_DATA['Target EBITDA (YTD)'] - US_MERGER_DATA['Target EBIT (YTD)']

# 5) IE = EBIT - Pre-Tax Income
US_MERGER_DATA['Target EBIT (YTD)'] = US_MERGER_DATA['Target EBIT (YTD)'] - US_MERGER_DATA['Target Pre-Tax Income (YTD)']

# 6) Tax = Pretax-Income - Net Income
US_MERGER_DATA['Target Pre-Tax Income (YTD)'] = US_MERGER_DATA['Target Pre-Tax Income (YTD)'] - US_MERGER_DATA['Target Net Income (YTD)']

US_MERGER_DATA.rename(columns={
    'Target  Net Sales (YTD)': 'Target Operating Expense (YTD)',
    'Target EBITDA (YTD)': 'Target Depreciation & Amortization (YTD)',
    'Target EBIT (YTD)': 'Target Interest Expense (YTD)',
    'Target Pre-Tax Income (YTD)': 'Target Tax (YTD)',
    'Target Net Asset': 'Target Total Liabilities'
},  inplace=True)

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

# X1 = X[non_categorical_column_names].dropna()

# print(X1.mean())
# X1 = np.asarray(X1)
# print(np.mean(X1, axis=0))
# print(np.var(X1, axis=0))

# print(X1.mean())
# print(X1.cov())
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

# X[non_categorical_column_names] = X[non_categorical_column_names].apply(np.exp)
# X.to_csv("C:/Users/kevin/Desktop/US_Merger_Data_Imputed2.csv")
# X = np.asarray(X)
# X_missing = np.asarray(X_missing)
# Y = np.asarray(Y)

labels = {
    "Completed": "Success",
    "Part Comp": "Success",
    "Withdrawn": "Failure"
}

# Ys = np.str((len(Y), ))
for i in range(len(Y)):
    Y[i] = labels[Y[i]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
del X, Y

skf = StratifiedKFold(n_splits=10)

# I. Gaussian Naive Bayes
NB_scores = list()  # score list for Naive Bayes
for train_indices, val_indices in skf.split(X_train, Y_train):
    # X_tr, X_val = X_train[train_indices], X_train[val_indices]
    # Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]

    ngb = GaussianNB()
    ngb.fit(X_train[train_indices], Y_train[train_indices])
    NB_scores.append(roc_auc_score(y_true=Y_train[val_indices], y_score=ngb.predict(X_train[val_indices])))
print(NB_scores)


# II. Logistic Regression with Elastic Net Regularization + SGD
LGR_scores = list()  # score list for Logistic Regression
for train_indices, val_indices in skf.split(X_train, Y_train):
    # X_tr, X_val = X_train[train_indices], X_train[val_indices]
    # Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]

    lgr = SGDClassifier(loss="log", penalty="elasticnet")
    lgr.fit(X_train[train_indices], Y_train[train_indices])
    LGR_scores.append(roc_auc_score(y_true=Y_train[val_indices], y_score=lgr.predict(X_train[val_indices])))
print(LGR_scores)

# # III SVM
# SVC_scores = list()  # score list for Logistic Regression
# for train_indices, val_indices in skf.split(X_train, Y_train):
#     X_tr, X_val = X_train[train_indices], X_train[val_indices]
#     Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]
#
#     svc = SVC()
#     svc.fit(X_tr, Y_tr)
#     SVC_scores.append(roc_auc_score(y_true=Y_val, y_score=svc.predict(X_val)))
# print(SVC_scores)

# III. Local GP
LGPC = LocalGaussianProcessClassifier(kernel_hyperparams=None, kernel_type="Matern", k=100,
                                      custom_kernel=Sum(Product(ConstantKernel(), Matern()), WhiteKernel()))
LGPC_scores = LGPC_CV(LGPC, score_criteria=roc_auc_score, n_splits=10).run_cv(X=X_train, Y=Y_train)

print(LGPC_scores)

