import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


MAX_FIT_VALUES = 5000
MAX_PLOT_VALUES = 1000
DEFAULT_CANDIDATES = (
    "normal", "lognormal", "weibull", "exponential", "gamma",
    "logistic", "gumbel_l", "gumbel_r",
)


@dataclass(frozen=True)
class DistributionSpec:
    scipy_dist: Any
    shape_names: tuple[str, ...] = ()
    output_loc: str = "loc"
    output_scale: str = "scale"
    positive_two_parameter: bool = False


FIT_SPECS = {
    "normal": DistributionSpec(stats.norm, output_loc="mean", output_scale="std"),
    "lognormal": DistributionSpec(stats.lognorm, ("shape",), positive_two_parameter=True),
    "weibull": DistributionSpec(stats.weibull_min, ("shape",), positive_two_parameter=True),
    "weibull-3p": DistributionSpec(stats.weibull_min, ("shape",)),
    "exponential": DistributionSpec(stats.expon, positive_two_parameter=True),
    "exponential-2p": DistributionSpec(stats.expon),
    "gamma": DistributionSpec(stats.gamma, ("shape",), positive_two_parameter=True),
    "logistic": DistributionSpec(stats.logistic),
    "gumbel_l": DistributionSpec(stats.gumbel_l),
    "gumbel_r": DistributionSpec(stats.gumbel_r),
}


def deterministic_limit(values: np.ndarray, maximum: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=float)
    if len(values) <= maximum:
        return values, False
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices], True


def named_parameters(name: str, fitted: tuple[float, ...]) -> dict[str, float]:
    spec = FIT_SPECS[name]
    shape_count = len(spec.shape_names)
    result = {key: float(value) for key, value in zip(spec.shape_names, fitted[:shape_count])}
    result[spec.output_loc] = float(fitted[-2])
    result[spec.output_scale] = float(fitted[-1])
    return result


def _fit_tuple(name: str, values: np.ndarray, fixed_loc: bool) -> tuple[float, ...]:
    spec = FIT_SPECS[name]
    kwargs = {"floc": 0.0} if fixed_loc else {}
    fitted = tuple(float(value) for value in spec.scipy_dist.fit(values, **kwargs))
    if not all(math.isfinite(value) for value in fitted) or fitted[-1] <= 0:
        raise ValueError("El ajuste produjo parámetros no válidos.")
    return fitted


def anderson_darling(values: np.ndarray, scipy_dist: Any, fitted: tuple[float, ...]) -> float:
    ordered = np.sort(values)
    probabilities = np.asarray(scipy_dist.cdf(ordered, *fitted), dtype=float)
    epsilon = np.finfo(float).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    indices = np.arange(1, len(ordered) + 1, dtype=float)
    terms = (2.0 * indices - 1.0) * (
        np.log(probabilities) + np.log1p(-probabilities[::-1])
    )
    return float(-len(ordered) - np.sum(terms) / len(ordered))


def _chi_square_gof(values: np.ndarray, scipy_dist: Any, fitted: tuple[float, ...], parameter_count: int) -> dict:
    max_bins = min(10, len(values) // 5)
    if max_bins < 4:
        return {"available": False, "reason": "Se requieren al menos 20 datos para frecuencias esperadas de 5."}
    bins = max_bins
    probabilities = np.linspace(0.0, 1.0, bins + 1)
    edges = np.asarray(scipy_dist.ppf(probabilities, *fitted), dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    if np.any(np.diff(edges) <= 0):
        return {"available": False, "reason": "Los cuantiles ajustados no producen intervalos distintos."}
    observed, _ = np.histogram(values, bins=edges)
    expected = len(values) * np.diff(scipy_dist.cdf(edges, *fitted))
    degrees_freedom = bins - 1 - parameter_count
    if degrees_freedom < 1 or np.any(expected < 5 - 1e-9):
        return {"available": False, "reason": "No hay grados de libertad o frecuencias esperadas suficientes."}
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    return {
        "available": True,
        "statistic": statistic,
        "p_value": float(stats.chi2.sf(statistic, degrees_freedom)),
        "degrees_freedom": int(degrees_freedom),
        "bins": int(bins),
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "method": "Intervalos definidos por cuantiles ajustados.",
    }


def fit_distribution(
    values: np.ndarray,
    name: str,
    bootstrap_reps: int = 200,
    seed: int = 1729,
) -> dict:
    spec = FIT_SPECS[name]
    values, sampled = deterministic_limit(np.asarray(values, dtype=float), MAX_FIT_VALUES)
    fixed_loc = bool(spec.positive_two_parameter and np.all(values > 0))
    fitted = _fit_tuple(name, values, fixed_loc)
    logpdf = np.asarray(spec.scipy_dist.logpdf(values, *fitted), dtype=float)
    if not np.all(np.isfinite(logpdf)):
        raise ValueError("El modelo asigna probabilidad nula a uno o más datos.")
    loglikelihood = float(np.sum(logpdf))
    parameter_count = len(fitted) - (1 if fixed_loc else 0)
    n = len(values)
    ad_statistic = anderson_darling(values, spec.scipy_dist, fitted)
    ks = stats.kstest(values, spec.scipy_dist.cdf, args=fitted)

    rng = np.random.default_rng(seed)
    exceedances = 0
    successful = 0
    failures = 0
    for _ in range(bootstrap_reps):
        try:
            generated = np.asarray(spec.scipy_dist.rvs(*fitted, size=n, random_state=rng), dtype=float)
            bootstrap_fit = _fit_tuple(name, generated, fixed_loc)
            bootstrap_ad = anderson_darling(generated, spec.scipy_dist, bootstrap_fit)
            successful += 1
            exceedances += bootstrap_ad >= ad_statistic
        except Exception:
            failures += 1
    ad_p_value = (exceedances + 1.0) / (successful + 1.0) if successful else None

    support = spec.scipy_dist.support(*fitted)
    return {
        "fitted_parameters": named_parameters(name, fitted),
        "sample_size": int(n),
        "source_sampled": sampled,
        "source_limit": MAX_FIT_VALUES,
        "parameter_count": int(parameter_count),
        "loglikelihood": loglikelihood,
        "aic": float(2 * parameter_count - 2 * loglikelihood),
        "bic": float(math.log(n) * parameter_count - 2 * loglikelihood),
        "anderson_darling": {
            "statistic": ad_statistic,
            "p_value": ad_p_value,
            "bootstrap_reps_requested": int(bootstrap_reps),
            "bootstrap_reps_successful": int(successful),
            "bootstrap_fit_failures": int(failures),
            "seed": int(seed),
        },
        "kolmogorov_smirnov": {
            "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "warning": "El p-value es aproximado porque los parámetros fueron estimados con los mismos datos.",
        },
        "chi_square": _chi_square_gof(values, spec.scipy_dist, fitted, parameter_count),
        "support": [float(support[0]), float(support[1])],
        "fit_policy": (
            f"MLE con loc=0; parametrización de {parameter_count} parámetros sobre soporte positivo."
            if fixed_loc else "MLE con loc libre."
        ),
        "_fitted_tuple": fitted,
    }


def identify_distributions(
    values: np.ndarray,
    candidates: list[str],
    bootstrap_reps: int,
    seed: int = 1729,
) -> dict:
    ranked = []
    failures = []
    for index, candidate in enumerate(candidates):
        try:
            result = fit_distribution(values, candidate, bootstrap_reps, seed + index)
            result.pop("_fitted_tuple", None)
            ranked.append({"distribution": candidate, **result})
        except Exception as error:
            failures.append({"distribution": candidate, "reason": str(error)})
    ranked.sort(key=lambda item: (item["anderson_darling"]["statistic"], item["aic"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return {
        "best_distribution": ranked[0]["distribution"] if ranked else None,
        "candidates": ranked,
        "failures": failures,
        "ranking_method": "Anderson-Darling ascendente, con AIC como desempate.",
    }


def probability_plot(values: np.ndarray, name: str, confidence: float) -> dict:
    values, sampled = deterministic_limit(np.asarray(values, dtype=float), MAX_PLOT_VALUES)
    values = np.sort(values)
    spec = FIT_SPECS[name]
    fixed_loc = bool(spec.positive_two_parameter and np.all(values > 0))
    fitted = _fit_tuple(name, values, fixed_loc)
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=float)
    plotting_probabilities = (ranks - 0.375) / (n + 0.25)
    theoretical = np.asarray(spec.scipy_dist.ppf(plotting_probabilities, *fitted), dtype=float)
    alpha = 1.0 - confidence
    lower_p = stats.beta.ppf(alpha / 2.0, ranks, n + 1.0 - ranks)
    upper_p = stats.beta.ppf(1.0 - alpha / 2.0, ranks, n + 1.0 - ranks)
    lower = np.asarray(spec.scipy_dist.ppf(lower_p, *fitted), dtype=float)
    upper = np.asarray(spec.scipy_dist.ppf(upper_p, *fitted), dtype=float)
    points = [
        {"theoretical": float(expected), "observed": float(observed)}
        for expected, observed in zip(theoretical, values)
    ]
    band = [
        {"theoretical": float(expected), "lower": float(low), "upper": float(high)}
        for expected, low, high in zip(theoretical, lower, upper)
    ]
    return {
        "fitted_parameters": named_parameters(name, fitted),
        "points": points,
        "confidence_band": band,
        "reference_line": {"slope": 1.0, "intercept": 0.0},
        "confidence": float(confidence),
        "sample_size": int(n),
        "source_sampled": sampled,
        "source_limit": MAX_PLOT_VALUES,
        "fit_policy": (
            f"MLE con loc=0; parametrización de {len(fitted) - 1} parámetros sobre soporte positivo."
            if fixed_loc else "MLE con loc libre."
        ),
        "warning": "La banda es puntual para cada estadístico de orden y no es una banda simultánea; no incorpora incertidumbre del ajuste.",
    }
