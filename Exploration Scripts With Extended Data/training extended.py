import pandas as pd
import numpy as np
from Modules.Models.LocalGP import LGPC_CV, LocalGaussianProcessClassifier
from sklearn.gaussian_process.kernels import (Matern, RationalQuadratic,
                                              ExpSineSquared, RBF, ConstantKernel,
                                              Product, Sum, WhiteKernel)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from Modules.UsefulFunctions import multivariate_gaussian_bayesian_imputation, multivariate_gaussian_bayesian_estimation


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


### Training + CV ...
X = US_MERGER_DATA.dropna().drop(["Status"], axis=1)
Y = US_MERGER_DATA.dropna()["Status"]
del US_MERGER_DATA

print(X.shape, Y.shape)
labels = {
    "Completed": 1,
    "Withdrawn": -1
}

# Ys = np.str((len(Y), ))
for i in range(len(Y)):
    Y[i] = labels[Y[i]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
del X, Y

skf = StratifiedKFold(n_splits=10)

# I. Gaussian Naive Bayes
NB_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    # X_tr, X_val = X_train[train_indices], X_train[val_indices]
    # Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]

    ngb = GaussianNB()
    ngb.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    NB_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
    NB_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
    NB_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
print(NB_scores)
del NB_scores

# II. Logistic Regression with Elastic Net Regularization + SGD
LGR_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    # X_tr, X_val = X_train[train_indices], X_train[val_indices]
    # Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]

    lgr = SGDClassifier(loss="log", penalty="elasticnet")
    lgr.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    LGR_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
    LGR_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
    LGR_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
print(LGR_scores)
del LGR_scores

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
# LGPC = LocalGaussianProcessClassifier(kernel_hyperparams=None, kernel_type="Matern", k=100,
#                                       custom_kernel=Sum(Product(ConstantKernel(), Matern()), WhiteKernel()))
# LGPC_scores = LGPC_CV(LGPC, score_criteria=roc_auc_score, n_splits=10).run_cv(X=X_train, Y=Y_train)
#
# print(LGPC_scores)

# III. GMM
GMM_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    GMM_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
    GMM_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
    GMM_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
print(GMM_scores)
del GMM_scores

# IV. Bayesian GMM
BGMM_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    bgmm = BayesianGaussianMixture(n_components=2)
    bgmm.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    BGMM_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
    BGMM_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
    BGMM_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
print(BGMM_scores)
del BGMM_scores