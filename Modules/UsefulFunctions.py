from numpy import log, sqrt, mean, sum, asarray, dot
from numpy.linalg import inv, pinv
from scipy.stats import invwishart, multivariate_normal
import pandas as pd


def mle_paramter_estimation(X, dis_typ):
    """
    :param X:
    :param dis_typ: str for distribution type
    :return:
    """
    X = asarray(X).reshape(-1,)

    if dis_typ == "Gamma":
        x_bar = mean(X)
        s = log(x_bar) - (1 / len(X)) * sum(log(X))
        alpha = (3 - s + sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        beta = alpha / x_bar
        return alpha, beta
    elif dis_typ == "LogNormal":
        logX = log(X)
        mu = mean(logX)
        sigma_sqrd = mean((logX - mu) ** 2)
        return mu, sigma_sqrd
    elif dis_typ == "Normal":
        mu = mean(X)
        sigma_sqrd = mean((X - mu) ** 2)
        return mu, sigma_sqrd
    else:
        raise DistributionTypeError("No such distribution supported")


def multivariate_gaussian_parameter_estimation(X):
    """
    X should be a pandas DataFrame
    :param X:
    :return:
    """
    return X.mean(), X.cov()


def multivariate_gaussian_bayesian_estimation(X, estimation_method="expectation",
                                              m=None, mu_0=None, sigma_0=None,
                                              psi=None, nu_0=None):
    """
    Returns posterior distribution parameters for mean and covariance for Multivariate Gaussian

    ----- For scalar prior hyper-parameters (m and nu_0) -----
    If only one of m or nu_0 is None, then set them equal to each other.
    If both are None, use n = number of samples w/out null values for m and nu_0.
    If both are satisfied, ... well make sure to use good priors

    ----- For vector & matrix prior hyper-parameters (mu_0, sigma_0, and psi) -----
    If any of them are not customized, impute the from the data

    :param X: Input Data in Pandas
    :param estimation_method: str
        "expectation": takes expectation of the parameters
        "MAP": takes MAP
        "random_sample": Take a random sample from the posterior distribution
        if None, just use expectation
    ----- Mean Prior Hyperparams (Model With Multivariate Gaussian) -----
    :param m: Number of samples to use to estimate mu_0 for prior (sample mean = mu_0)
    :param mu_0: (Mean of Mean) If not None, then use customized priors
    :param sigma_0: (Cov of Mean) If not none, use custom priors

    ----- Covariance Prior Hyperparams (Model With Inverse-Wishart Distribution) -----
    :param psi: (Scale) If not none, use custom priors
    :param nu_0: (Degree of Freedom) number of samples to use to estimate sigma_0 = (1/m)*sigma & Psi = nu_0*sigma_0 such that sample mean = mu_0

    :return: posterior mean (d x 1) and sigma (d x d)
    """
    X = X.dropna()
    n, d = X.shape  # number of samples, dimensionality

    # I: Derive Posterior
    # -> Deal with unspecified m and nu_0
    if m and nu_0: # both m and nu_0 are specified
        pass
    elif not m and not nu_0: # neither are specified
        m = n
        nu_0 = n
    elif nu_0: # only nu_0 is specified
        m = nu_0
    elif m:  # only m is specified
        nu_0 = m

    # print(n, m, nu_0)
    # -> Deal with No custom Priors
    # TODO: X.sample(n=m) or X.sample(n=nu_0) ????
    if not mu_0 and not sigma_0:  # both mu_0 and sigma_0 are not specified
        mu_0, sigma_0 = multivariate_gaussian_parameter_estimation(X.sample(n=m))
    elif not mu_0:  # only mu_0 is not specified
        mu_0 = X.sample(n=m).mean()
    elif not sigma_0:  # only sigma_0 is not specified
        sigma_0 = X.sample(n=m).cov()
    if not psi:
        psi = nu_0 * sigma_0

    # print("mu", mu_0)
    # print("sigma", sigma_0)
    # print("psi", psi)

    # Find Posterior Distribution Parameters
    x_bar, S = multivariate_gaussian_parameter_estimation(X)

    # Mu: Mean, Cov
    mu_posterior_params = (
        (n * x_bar + m * mu_0) / (n + m),
        sigma_0 / (n + m)
    )
    # Sigma: Scale, df
    a = asarray(x_bar - mu_0).reshape(-1, 1)
    sigma_posterior_params = (
        psi + n * S + ((n * m) / (n + m)) * dot(a, a.T),
        n + nu_0
    )

    # II. Estimation
    if estimation_method == "MAP":
        return asarray(mu_posterior_params[0]).reshape(-1, 1), \
               asarray(sigma_posterior_params[0]) / (sigma_posterior_params[1] + d + 1)
    elif estimation_method == "random_sample":
        return multivariate_normal.rvs(mean=mu_posterior_params[0], cov=mu_posterior_params[1]).reshape(-1, 1), \
               invwishart.rvs(scale=sigma_posterior_params[0], df=sigma_posterior_params[1])
    else:
        return asarray(mu_posterior_params[0]).reshape(-1, 1), \
               asarray(sigma_posterior_params[0]) / (sigma_posterior_params[1] - d + 1)


def multivariate_gaussian_bayesian_imputation(X, mu, sigma, imputation_method="expectation"):
    # TODO: Test this function
    """
    :param X: Data in Pandas with missing data points
    :param mu: Mean vector of multivariate gaussian
    :param sigma: Cov matrix of multivariate gaussian
    :param imputation_method:
        "expectation": takes expectation
        "MAP": takes MAP
        "random_sample": Take a random sample from the posterior distribution
        if None, just use expectation
    :return: Data with imputed missing values
    """

    # I: Convert parameters to pandas data frame
    mu = pd.DataFrame.from_records(data=mu.reshape(1, -1), columns=X.columns)
    sigma = pd.DataFrame.from_records(data=sigma, columns=X.columns)
    names = dict()
    i = 0
    for col in X.columns:
        names[i] = col
        i += 1
    sigma.rename(index=names, inplace=True)

    # TODO: Implementing conditional distribution?
    for i, row in X.iterrows():
        nulls = row.isna()
        null_cols = X.columns[nulls]
        available_cols = X.columns[~nulls]

        available_values = asarray(X.loc[i, available_cols]).reshape(-1, 1)

        # conditional multivariate gaussian parameters

        sigma_12 = sigma.loc[null_cols, available_cols]
        sigma_22 = sigma.loc[available_cols, available_cols]

        mu_cond = mu[null_cols] + dot(dot(sigma_12, inv(sigma_22)), asarray(available_values - mu[available_cols].T) ).T
        if imputation_method == "random_sample":
            sigma_11 = sigma.loc[null_cols, null_cols]
            sigma_21 = sigma.loc[available_cols, null_cols]

            sigma_cond = sigma_11 - sigma_12@inv(sigma_22)@sigma_21
            X.at[i, null_cols] = multivariate_normal.rvs(mean=mu_cond, cov=sigma_cond)
        else:
            for col in null_cols:
                X.at[i, col] = mu_cond[col]


