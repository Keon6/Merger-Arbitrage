import pandas as pd
import numpy as np
from sklearn.cluster import k_means, dbscan
from Modules.UsefulFunctions import multivariate_gaussian_mle_estimation
from plotly.offline import plot
import plotly.graph_objs as go

us_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/US_Merger_Data_Scrubbed_No_DefaultDistance.csv"
intl_data_path = "C:/Users/kevin/OneDrive/Desktop/RISK ARBITRAGE/SDC/Intl_Merger_Data_Scrubbed_No_DefaultDistance.csv"

US_MERGER_DATA = pd.read_csv(us_data_path)
print(US_MERGER_DATA.head())
print(US_MERGER_DATA.shape)  # (121965, 270)
print("---- Null Count ----")
print(US_MERGER_DATA.isnull().sum())
print("---- Column Names and types ----")
colnames = US_MERGER_DATA.columns
for col in colnames:
    print("'" + col+"'", US_MERGER_DATA[col].dtypes)

# I. Explore How the Data is Distributed
numerical_cols = US_MERGER_DATA.columns[3:30]
# i = 3
# for col in numerical_cols:
#     if i <= 5:
#         X = US_MERGER_DATA[col]
#     else:
#         X = np.log(US_MERGER_DATA[col])
#     a = col.replace("/", " per ")
#     data = [go.Histogram(
#         x=X,
#         histnorm='probability'
#     )]
#     layout = go.Layout(
#         title=col
#     )
#     plot(data, filename=f"{a}")
#     i += 1

# II. Transform Numerical Data to multivariate Gaussian

# III. Estimate parameters of multivariate Gaussian and do inference
Mu = US_MERGER_DATA[numerical_cols].mean()
Sigma = US_MERGER_DATA[numerical_cols].cov()

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


