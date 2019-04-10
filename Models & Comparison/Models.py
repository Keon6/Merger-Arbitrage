import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Scrubbed_No_DefaultDistance.csv"
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_Scrubbed_No_DefaultDistance.csv"


US_MERGER_DATA = pd.read_csv(us_data_path)
real_valued_cols = US_MERGER_DATA.columns[3:30]
numerical_cols = US_MERGER_DATA.columns[3:]

Y = US_MERGER_DATA["Status"]
X = US_MERGER_DATA.drop(["Status"], axis=1)

# PCA for some models
pca = PCA()
X_pca = pca.fit_transform(X)


# Data Split for CV
kf = KFold(n_splits=10)
NB_scores = list()  # score list for Naive Bayes
LGR_scores = list()  # score list for Logistic Regression
# skf = StratifiedKFold(n_splits=10)
for train_indices, test_indices in kf.split(X):
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    X_pca_train, X_pca_test = X_pca[train_indices], X_pca[test_indices]
    Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]

    # I. PCA + Gaussian Naive Bayes
    ngb = GaussianNB()
    ngb.fit(X_pca_train, Y_train)
    NB_scores.append(roc_auc_score(y_true=Y_test, y_score=ngb.predict(X_pca_test)))

    # II. Logistic Regression with Elastic Net Regularization + SGD
    lgr = SGDClassifier(loss="log", penalty="elasticnet")
    lgr.fit(X_train, Y_train)
    LGR_scores.append(roc_auc_score(y_true=Y_test, y_score=lgr.predict(X_pca_test)))

    # III.



