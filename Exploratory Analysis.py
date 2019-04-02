import pandas as pd
import numpy as np
from sklearn.cluster import k_means, dbscan
import plotly
from plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go

us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Scrubbed_No_DefaultDistance.csv"
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_Scrubbed_No_DefaultDistance.csv"

US_MERGER_DATA = pd.read_csv(us_data_path,)
print(US_MERGER_DATA.head())
print(US_MERGER_DATA.shape)  # (121965, 270)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())

# I. Explore How the Data is Distributed
for col in US_MERGER_DATA.columns[6:20]:
    a = col.replace("/", " per ")
    data = [go.Histogram(
        x=np.log(US_MERGER_DATA[col])
    )]
    layout = go.Layout(
        title=col
    )
    plot(data, filename=f"{a}")

# Observation from Histogram:
# Status: Significantly more success than failure (TODO: perhaps SDC is missing lots of failure data? or not?)
# % of Shares Owned, Transacted, Sought, ... tend to be around 100% but a few below
# Transaction Value  (negative log shaped (right side of the bell curve)
#   - Significantly more deals as deal sizes gets smaller
#   - Pretty Steep Decline
#   - Possible Distributions to try:
#       -> Gamma with low alpha(a<=1)?
#       -> Folded normal Distribution ?
#       -> Log Normal?
#       -> Inverse Normal?
# Target Debt: //
# Target Enterprise Value: //
# Target Equity Value //
# Target Total Asset: //
# Target/Acquiror Total Advisor Fee: //
# Net Sales: few negative values possible, otherwise //
# Target EBIT: negative Values possible, right skewed, otherwise //
# Target EBITDA //
# Target Net Asset: //
# Target Common Equity: //
#
# Target Pre-tax Income: Approximately 0 centered probably little to right), right skewed (slightly).
# Target Net Income: //
# Target Book Val/Share : approximately normal, really skinny
# Target EPS: approximately normal be left skweed
# Premiums: Right Skewed, mostly 0~25 % and some negative and above 100%

# Price/Share:  -> Probably not that important
#

# II: Dealing With Missing Data
# NOTE: We believe that the missing data isn't missing at random. Because the nature of M&As suggest that
# some information may not be released due to a reason that depends on some of the other data attributes
# (both observed and unobserved).
# Therefore, imputing missing data seems necessary.

# IDEAS:
# 1. MLE/MAP estimation of parameters for model distributions & random sample
# 2. kNN Imputation
# 2. GP Imputation
# 3. Regression Imputation
# 4. "Bayesian" Imputation: Find joint density f(x1, x2, ..., xd). for missing points, find
# E[f(x1, ..., xd)|x1=a1, x2=a2, ..., xd=ad]


# MLE estimates
# 1) Gamma dist
# beta = alpha*n/sum(x_i, i=1->n)
# intermediate term s = ln(mean(x)) - (1/n)sum(ln(x_i),i=1->n)
# alpha =(approx) (3-s + sqrt((s-3)**2 + 24s))/12s

# 2) LogNormal
# mu = sum(ln(x_i), i=1->n)
# sigma^2 = sum((ln(x_i)-mu)**2, i=1->n)


def mle_parameters(X, distribution):
    """
    :param X:
    :param distribution:
    :return:
    """
    X = np.asarray(X).reshape(-1,)
    if distribution == "Gamma":
        x_bar = np.mean(X)
        s = np.log(x_bar) - (1 / len(X)) * np.sum(np.log(X))
        alpha = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        beta = alpha / x_bar
        return alpha, beta
    elif distribution == "LogNormal":
        logX = np.log(X)
        mu = np.mean(logX)
        sigma_sqrd = np.mean((logX - mu) ** 2)
        return mu, sigma_sqrd
    elif distribution == "Normal":
        mu = np.mean(X)
        sigma_sqrd = np.mean((X - mu) ** 2)
        return mu, sigma_sqrd
    else:
        pass




