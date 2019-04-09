from numpy import asarray, log, sqrt, mean, sum, nanmean
import numpy.ma as ma
from Modules.Errors import DistributionTypeError


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


def multivariate_gaussian_mle_estimation(X):
    return ma.mean(X), ma.cov(X)