from __future__ import annotations

import warnings

import numpy as np
from scipy import special
from scipy.stats import gamma, genhyperbolic, geninvgauss, norminvgauss

warnings.filterwarnings("ignore")


def Algeb(x, A, B):
    return A * x / np.sqrt(1 + B * x**2)


def Erf_PL(x, A, B):
    return special.erf(A * (x**B)) * 100


def Exp_PL(x, A, B):
    return np.exp(-A * (x**B))


def Gamma(x, A, B):
    return B * gamma.cdf(x, A) * 100


def GLH_p05(x, A, B):
    return genhyperbolic.cdf(x, p=-0.5, a=A, b=B) * 100


def GLH_p1(x, A, B):
    return genhyperbolic.cdf(x, p=1, a=A, b=B) * 100


def Tanh(x, A, B):
    return 0.5 * (1 + np.tanh((x - A) / B)) * 100


def Log_exp(x, A, B):
    return A * np.exp(B * np.log10(x))


def Ln(x, A, B):
    return A * np.log(x) + B


def Lognormal(x, A, B):
    return 0.5 * (1 + special.erf((np.log(x) - B) / (A * np.sqrt(2)))) * 100


def LLaplace(x, A, B):
    signed = np.sign(np.log(x) - B)
    return 1 / (2 * A) * (1 + signed * (1 - np.exp(-np.absolute(np.log(x) - B) / A))) * 100


def NIG(x, a, b):
    return norminvgauss.cdf(x, a, b) * 100


def PL(x, A, B):
    return A * (x ** (-B))


def GIG_2p(x, A, B):
    return geninvgauss.cdf(x, A, B) * 100


def Weibull(x, A, B):
    return (1 - np.exp(-((x / A) ** B))) * 100


def GLH(x, p, a, b):
    return genhyperbolic.cdf(x, p, a, b) * 100


def Logn_Weib(x, A, B, C):
    return (1 + special.erf((np.log(x) - B) / (A * np.sqrt(2)))) * (1 - np.exp(-(x**C))) * 100


def Logn_PL(x, A, B, C):
    return 0.5 * (1 + special.erf((np.log(x) - B) / (A * np.sqrt(2)))) * (x ** (-C)) * 100


def Logn_Tanh(x, A, B, C):
    return (1 + special.erf((np.log(x) - B) / (A * np.sqrt(2)))) * (1 + np.tanh(x - C)) * 100


def LSLaplace(x, A, B, C):
    signed = np.sign(np.log(x) - C)
    exponent = -np.absolute(np.log(x) - C) * 2 / (A + B - signed * (A - B))
    inner = 0.5 * (signed * (A + B) - A + B) / (A + B) * (1 - np.exp(exponent))
    return A / (A + B) * (1 + inner) * 100


def PL_Tanh(x, A, B, C):
    return A * (x ** (-B)) * (1 + np.tanh(x - C)) * 100


def pPL_Exp(x, A, B, C):
    return A * (x**B) * np.exp(-(x / C))


def PL_omExp(x, A, B, C):
    return A * (x**B) * (1 - np.exp(-(x / C))) * 100


def PL_Weib(x, A, B, C):
    return A * (x ** (-B)) * (1 - np.exp(-(x**C))) * 100


def GIG_3p(x, A, B, C):
    return geninvgauss.cdf(x, A, B, loc=0, scale=C) * 100

