import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, average_precision_score
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import scale

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.metrics import SCORERS


US_MERGER_DATA = pd.read_csv("C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Imputed.csv",
                             index_col=0)
# print(sorted(SCORERS.keys()))
for col in US_MERGER_DATA.columns:
    if US_MERGER_DATA[col].isnull().sum() != 0:
        print(col, US_MERGER_DATA[col].isnull().sum())
print(US_MERGER_DATA.shape)
#
US_MERGER_DATA = US_MERGER_DATA.sample(n=37000)
# Input Data and Label Split + Train/Test Split
X = US_MERGER_DATA.drop(["Status", " Target Goodwill"], axis=1)
Y = US_MERGER_DATA["Status"]
del US_MERGER_DATA

# print(X.shape, Y.shape)
labels = {
    "Completed": 1,
    "Withdrawn": -1
}

# Ys = np.str((len(Y), ))
for i in range(len(Y)):
    Y.iloc[i] = labels[Y.iloc[i]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
del X, Y

skf = StratifiedKFold(n_splits=10)

#######################
# I. Gaussian Naive Bayes

NB_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):

    ngb = GaussianNB()
    ngb.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    NB_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
    NB_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
    NB_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], ngb.predict(X_train.iloc[val_indices])))
print(NB_scores)
for key in NB_scores:
    print(key, np.mean(NB_scores[key]))
del NB_scores

# II. Logistic Regression with Elastic Net Regularization + SGD


# Bayesian hyperparmeter opt
lgr = SGDClassifier(loss="log", penalty="elasticnet", max_iter=1000,)
space = [
    Real(0.0001, 1, name='alpha'),
    Real(0, 1, name='l1_ratio')
]
#
# lgr = LogisticRegression(max_iter=100)
# space = [
#     Categorical(["l1", "l2"], name="penalty"),
#     Real(0.001, 10000, name='C')
# ]

@use_named_args(space)
def objective_lgr(**params):
    lgr.set_params(**params)
    return np.mean(cross_val_score(lgr, X_train, Y_train, cv=10, n_jobs=-1, scoring="average_precision"))

res_gp_lgr = gp_minimize(objective_lgr, space, n_calls=100, random_state=0)
print("---- LGR ----")
print("Best Score:", res_gp_lgr.fun)
print("Best Hyper-Parameters:", res_gp_lgr.x)
del lgr

# CV scores for others using optimal hyper-parameters
LGR_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
#
# lgr = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.9993730173334414, l1_ratio=0.006537148124680538)
for train_indices, val_indices in skf.split(X_train, Y_train):
    # X_tr, X_val = X_train[train_indices], X_train[val_indices]
    # Y_tr, Y_val = Y_train[train_indices], Y_train[val_indices]
    # lgr = SGDClassifier(loss="log", penalty="elasticnet", alpha=res_gp_lgr.x[0], l1_ratio=res_gp_lgr.x[1])
    lgr = LogisticRegression(penalty=res_gp_lgr.x[0], C=res_gp_lgr.x[1])
    lgr.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    LGR_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
    LGR_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
    LGR_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], lgr.predict(X_train.iloc[val_indices])))
print(LGR_scores)
for key in LGR_scores:
    print(key, np.mean(LGR_scores[key]))
del LGR_scores, space, X_test, Y_test


# III. GMM
# Bayesian Hyperparam Opt
gmm = GaussianMixture(n_components=2)
space = [
    Categorical(['full', 'tied', 'diag', 'spherical'], name='covariance_type'),
    Real(1.0**-6, 10.0, name='reg_covar')
]

@use_named_args(space)
def objective_gmm(**params):
    gmm.set_params(**params)
    return np.mean(cross_val_score(gmm, X_train, Y_train, cv=10, n_jobs=-1, scoring="average_precision"))


res_gp_gmm = gp_minimize(objective_gmm, space, n_calls=200, random_state=0)
print("---- GMM ----")
print("Best Score:", res_gp_gmm.fun)
print("Best Hyper-Parameters:", res_gp_gmm.x)
del gmm

# gmm = GaussianMixture(n_components=2, covariance_type=res_gp_gmm.x[0], reg_covar=res_gp_gmm.x[1])
#
GMM_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    gmm = GaussianMixture(n_components=2, covariance_type=res_gp_gmm.x[0], reg_covar=res_gp_gmm.x[1])
    # gmm = GaussianMixture(n_components=2, covariance_type='spherical', reg_covar=10)
    gmm.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    GMM_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
    GMM_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
    # GMM_scores["PREC"].append(precision_score(Y_train.iloc[val_indices], gmm.predict(X_train.iloc[val_indices])))
print(GMM_scores)
for key in GMM_scores:
    print(key, np.mean(GMM_scores[key]))
del GMM_scores

# IV. Bayesian GMM
# Bayesian Hyperparam Opt
# bgmm = BayesianGaussianMixture(n_components=1)
# space = [
#     Categorical(['full', 'tied', 'diag', 'spherical'], name='covariance_type'),
#     Real(1.0**-6, 10.0, name='reg_covar'),
#     Categorical(['dirichlet_process', 'dirichlet_distribution'], name="weight_concentration_prior_type"),
#     Real(1.0**-6, 10.0, name="weight_concentration_prior"),
#     Real(1.0**-6, 10.0, name="mean_precision_prior"),
#     Integer(len(X_train.columns), 3*len(X_train.columns), name="degrees_of_freedom_prior")
# ]
#
# @use_named_args(space)
# def objective_bgmm(**params):
#     bgmm.set_params(**params)
#     return np.mean(cross_val_score(bgmm, X_train, Y_train, cv=10, n_jobs=-1, scoring="average_precision"))
#
#
# res_gp_bgmm = gp_minimize(objective_bgmm, space, n_calls=50, random_state=0)
# print("---- GMM ----")
# print("Best Score:", res_gp_bgmm.fun)
# print("Best Hyper-Parameters:", res_gp_bgmm.x)
# del bgmm

# bgmm = BayesianGaussianMixture(n_components=2, covariance_type=res_gp_bgmm.x[0], reg_covar=res_gp_bgmm.x[1],
#                                weight_concentration_prior_type=res_gp_bgmm.x[2],
#                                weight_concentration_prior=res_gp_bgmm.x[3],
#                                mean_precision_prior=res_gp_bgmm.x[4], degrees_of_freedom_prior=res_gp_bgmm.x[5])

BGMM_scores = {
    "AUC": [],
    "ACCU": [],
    "PREC": []
}
for train_indices, val_indices in skf.split(X_train, Y_train):
    # bgmm = BayesianGaussianMixture(n_components=2, covariance_type=res_gp_bgmm.x[0], reg_covar=res_gp_bgmm.x[1],
    #                                weight_concentration_prior_type=res_gp_bgmm.x[2],
    #                                weight_concentration_prior=res_gp_bgmm.x[3],
    #                                mean_precision_prior=res_gp_bgmm.x[4], degrees_of_freedom_prior=res_gp_bgmm.x[5])
    bgmm = BayesianGaussianMixture(n_components=2, covariance_type="spherical", reg_covar=1
                                   )
    bgmm.fit(X_train.iloc[train_indices], Y_train.iloc[train_indices])
    BGMM_scores["AUC"].append(roc_auc_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
    BGMM_scores["ACCU"].append(accuracy_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
    BGMM_scores["PREC"].append(average_precision_score(Y_train.iloc[val_indices], bgmm.predict(X_train.iloc[val_indices])))
print(BGMM_scores)
for key in BGMM_scores:
    print(key, np.mean(BGMM_scores[key]))
del BGMM_scores
