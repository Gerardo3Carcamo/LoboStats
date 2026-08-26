import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from utils.serialization import json_safe


def confidence_interval(values: pd.Series, confidence: float, statistic: str, resamples: int, seed: int) -> dict:
    estimate = float(values.mean() if statistic == "mean" else values.median())
    if statistic == "mean":
        sem = stats.sem(values)
        low, high = stats.t.interval(confidence, len(values) - 1, loc=estimate, scale=sem)
        method = "student_t"
    else:
        rng = np.random.default_rng(seed)
        samples = rng.choice(values.to_numpy(), size=(resamples, len(values)), replace=True)
        low, high = np.quantile(np.median(samples, axis=1), [(1 - confidence) / 2, (1 + confidence) / 2])
        method = "percentile_bootstrap"
    return json_safe({"statistic": statistic, "estimate": estimate, "confidence": confidence,
                      "lower": low, "upper": high, "sample_size": len(values), "method": method,
                      "resamples": resamples if statistic == "median" else None,
                      "seed": seed if statistic == "median" else None})


def one_sample_z(values: pd.Series, null_mean: float, population_std: float, alpha: float,
                 alternative: str) -> dict:
    sample_mean = float(values.mean())
    standard_error = population_std / np.sqrt(len(values))
    statistic = (sample_mean - null_mean) / standard_error
    if alternative == "larger":
        p_value = stats.norm.sf(statistic)
    elif alternative == "smaller":
        p_value = stats.norm.cdf(statistic)
    else:
        p_value = 2 * stats.norm.sf(abs(statistic))
    critical = stats.norm.ppf(1 - alpha / 2)
    return json_safe({
        "z_statistic": statistic,
        "p_value": p_value,
        "sample_mean": sample_mean,
        "null_mean": null_mean,
        "population_std": population_std,
        "standard_error": standard_error,
        "confidence_interval": [sample_mean - critical * standard_error, sample_mean + critical * standard_error],
        "sample_size": len(values),
        "alternative": alternative,
    })


def matrix(frame: pd.DataFrame, columns: list[str], method: str, covariance: bool = False) -> dict:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.cov() if covariance else numeric.corr(method=method)
    return json_safe({"columns": columns, "matrix": values.to_numpy(),
                      "pair_counts": numeric.notna().astype(int).T.dot(numeric.notna().astype(int)).to_numpy(),
                      "method": "covariance" if covariance else method})


def normality(values: pd.Series, method: str, alpha: float) -> dict:
    if method == "shapiro":
        result = stats.shapiro(values)
        statistic, p_value, critical = result.statistic, result.pvalue, None
    elif method == "anderson":
        result = stats.anderson(values, dist="norm")
        levels = np.asarray(result.significance_level) / 100
        index = int(np.argmin(np.abs(levels - alpha)))
        statistic, critical = result.statistic, result.critical_values[index]
        p_value = None
    elif method == "kolmogorov-smirnov":
        standardized = (values - values.mean()) / values.std(ddof=1)
        result = stats.kstest(standardized, "norm")
        statistic, p_value, critical = result.statistic, result.pvalue, None
    else:
        result = stats.normaltest(values)
        statistic, p_value, critical = result.statistic, result.pvalue, None
    rejected = bool(statistic > critical) if critical is not None else bool(p_value < alpha)
    return json_safe({"method": method, "statistic": statistic, "p_value": p_value,
                      "critical_value": critical, "alpha": alpha, "reject_normality": rejected,
                      "sample_size": len(values)})


def rank_correlation(x: pd.Series, y: pd.Series, method: str) -> dict:
    result = stats.spearmanr(x, y) if method == "spearman" else stats.kendalltau(x, y)
    return json_safe({"method": method, "coefficient": result.statistic, "p_value": result.pvalue,
                      "sample_size": len(x)})


def partial_correlation(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict:
    data = frame[[x, y, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
    design = np.column_stack([np.ones(len(data)), data[controls].to_numpy()])
    residual_x = data[x].to_numpy() - design @ np.linalg.lstsq(design, data[x], rcond=None)[0]
    residual_y = data[y].to_numpy() - design @ np.linalg.lstsq(design, data[y], rcond=None)[0]
    result = stats.pearsonr(residual_x, residual_y)
    degrees = len(data) - len(controls) - 2
    return json_safe({"coefficient": result.statistic, "p_value": result.pvalue, "sample_size": len(data),
                      "controls": controls, "degrees_of_freedom": degrees, "method": "partial_pearson"})


def proportion_one(successes: int, nobs: int, null: float, alpha: float, alternative: str) -> dict:
    statistic, p_value = proportions_ztest(successes, nobs, value=null, alternative=alternative)
    low, high = proportion_confint(successes, nobs, alpha=alpha, method="wilson")
    return json_safe({"z_statistic": statistic, "p_value": p_value, "proportion": successes / nobs,
                      "null_proportion": null, "confidence_interval": [low, high], "method": "z_test_wilson_ci"})


def proportion_two(successes: list[int], nobs: list[int], alpha: float, alternative: str) -> dict:
    statistic, p_value = proportions_ztest(successes, nobs, alternative=alternative)
    proportions = np.asarray(successes) / np.asarray(nobs)
    difference = proportions[0] - proportions[1]
    se = np.sqrt(sum(p * (1 - p) / n for p, n in zip(proportions, nobs)))
    critical = stats.norm.ppf(1 - alpha / 2)
    return json_safe({"z_statistic": statistic, "p_value": p_value, "proportions": proportions,
                      "difference": difference, "confidence_interval": [difference - critical * se, difference + critical * se],
                      "method": "two_proportion_z_test"})


def power_analysis(kind: str, effect_size: float, alpha: float, power: float | None, nobs: float | None,
                   alternative: str = "two-sided", groups: int = 2) -> dict:
    if kind == "t-test":
        analysis = TTestPower()
        solved = analysis.solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, alternative=alternative)
        key = "power" if power is None else "sample_size"
    elif kind == "t-test-independent":
        analysis = TTestIndPower()
        solved = analysis.solve_power(effect_size=effect_size, nobs1=nobs, alpha=alpha, power=power, alternative=alternative)
        key = "power" if power is None else "sample_size_per_group"
    else:
        analysis = FTestAnovaPower()
        solved = analysis.solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, k_groups=groups)
        key = "power" if power is None else "total_sample_size"
    return json_safe({key: solved, "effect_size": effect_size, "alpha": alpha, "groups": groups if kind == "anova" else None})
