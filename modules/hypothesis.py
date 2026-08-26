import numpy as np
import pandas as pd
from scipy import stats

from utils.serialization import json_safe


def decision(p_value: float, alpha: float, context: str) -> str:
    if p_value < alpha:
        return f"Se rechaza la hipótesis nula; hay evidencia estadísticamente significativa {context} (α={alpha:g})."
    return f"No se rechaza la hipótesis nula; no hay evidencia estadísticamente significativa {context} (α={alpha:g})."


def one_sample_t(values: pd.Series, popmean: float) -> dict:
    statistic, p_value = stats.ttest_1samp(values, popmean)
    return json_safe({"statistic": statistic, "p_value": p_value, "degrees_of_freedom": len(values) - 1, "sample_mean": values.mean(), "null_mean": popmean})


def independent_t(first: pd.Series, second: pd.Series) -> dict:
    statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
    first_var, second_var = first.var(ddof=1), second.var(ddof=1)
    numerator = (first_var / len(first) + second_var / len(second)) ** 2
    denominator = (first_var / len(first)) ** 2 / (len(first) - 1) + (second_var / len(second)) ** 2 / (len(second) - 1)
    degrees = numerator / denominator if denominator else np.nan
    return json_safe({
        "statistic": statistic, "p_value": p_value, "degrees_of_freedom": degrees,
        "first_mean": first.mean(), "second_mean": second.mean(),
        "first_size": len(first), "second_size": len(second), "equal_variance": False,
    })


def paired_t(first: pd.Series, second: pd.Series) -> dict:
    statistic, p_value = stats.ttest_rel(first, second)
    return json_safe({
        "statistic": statistic, "p_value": p_value, "degrees_of_freedom": len(first) - 1,
        "first_mean": first.mean(), "second_mean": second.mean(), "pair_count": len(first),
    })


def anova_test(groups: list[tuple[object, pd.Series]]) -> dict:
    statistic, p_value = stats.f_oneway(*(values for _, values in groups))
    total = sum(len(values) for _, values in groups)
    return json_safe({
        "statistic": statistic, "p_value": p_value,
        "degrees_of_freedom_between": len(groups) - 1,
        "degrees_of_freedom_within": total - len(groups),
        "groups": [{"name": name, "count": len(values), "mean": values.mean()} for name, values in groups],
    })


def chi_square_test(table: pd.DataFrame) -> dict:
    statistic, p_value, degrees, expected = stats.chi2_contingency(table)
    return json_safe({
        "statistic": statistic, "p_value": p_value, "degrees_of_freedom": degrees,
        "observed": table.to_numpy(), "expected": expected,
        "row_labels": table.index.tolist(), "column_labels": table.columns.tolist(),
    })


def correlation_test(x: pd.Series, y: pd.Series) -> dict:
    result = stats.pearsonr(x, y)
    return json_safe({"coefficient": result.statistic, "p_value": result.pvalue, "sample_size": len(x), "method": "pearson"})
