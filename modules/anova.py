import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from utils.serialization import dataframe_records, json_safe


def _table(frame: pd.DataFrame) -> list[dict]:
    output = frame.reset_index().rename(columns={"index": "term", "PR(>F)": "p_value", "F Value": "f_value",
                                                        "Num DF": "df_num", "Den DF": "df_den"})
    return dataframe_records(output)


def residual_diagnostics(model, alpha: float) -> dict:
    residuals = np.asarray(model.resid, dtype=float)
    result = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 and np.ptp(residuals) > 0 else None
    fitted = np.asarray(model.fittedvalues, dtype=float)
    return json_safe({"count": len(residuals), "mean": residuals.mean(), "std": residuals.std(ddof=1),
                      "shapiro": {"statistic": result.statistic, "p_value": result.pvalue,
                                  "is_normal": bool(result.pvalue >= alpha)} if result else None,
                      "fitted_residual_sample": [{"fitted": f, "residual": r} for f, r in zip(fitted[:500], residuals[:500])]})


def one_way(data: pd.DataFrame, alpha: float) -> dict:
    model = ols("value ~ C(group)", data=data).fit()
    table = anova_lm(model, typ=2)
    ss_between, ss_residual = table.loc["C(group)", "sum_sq"], table.loc["Residual", "sum_sq"]
    return json_safe({"table": _table(table), "eta_squared": ss_between / (ss_between + ss_residual),
                      "groups": dataframe_records(data.groupby("group", as_index=False)["value"].agg(["count", "mean", "std"])),
                      "residual_diagnostics": residual_diagnostics(model, alpha)})


def two_way(data: pd.DataFrame, interaction: bool, typ: int, alpha: float) -> dict:
    formula = "value ~ C(factor_a) * C(factor_b)" if interaction else "value ~ C(factor_a) + C(factor_b)"
    model = ols(formula, data=data).fit()
    return json_safe({"table": _table(anova_lm(model, typ=typ)), "typ": typ, "interaction": interaction,
                      "cell_means": dataframe_records(data.groupby(["factor_a", "factor_b"], as_index=False)["value"].agg(["count", "mean", "std"])),
                      "residual_diagnostics": residual_diagnostics(model, alpha)})


def post_hoc(data: pd.DataFrame, method: str, alpha: float) -> dict:
    groups = [(name, frame["value"].to_numpy()) for name, frame in data.groupby("group")]
    if method == "tukey":
        result = pairwise_tukeyhsd(data["value"], data["group"], alpha=alpha)
        rename = {"group_t": "group1", "group_c": "group2", "meandiff": "mean_difference", "p-adj": "p_value"}
        rows = [{rename.get(name, name): json_safe(value) for name, value in zip(result._results_table.data[0], row)}
                for row in result._results_table.data[1:]]
        return {"method": method, "comparisons": rows}
    pairs, raw = [], []
    model = ols("value ~ C(group)", data=data).fit()
    mse, df_error = float(np.sum(model.resid ** 2) / model.df_resid), int(model.df_resid)
    k = len(groups)
    for (name_a, a), (name_b, b) in itertools.combinations(groups, 2):
        difference = float(np.mean(a) - np.mean(b)); se = np.sqrt(mse * (1 / len(a) + 1 / len(b)))
        if method == "bonferroni":
            statistic = difference / se; p = 2 * stats.t.sf(abs(statistic), df_error); raw.append(p)
            pairs.append({"group1": name_a, "group2": name_b, "mean_difference": difference,
                          "statistic": statistic, "standard_error": se})
        else:
            f_stat = (difference / se) ** 2 / (k - 1)
            p = stats.f.sf(f_stat, k - 1, df_error)
            critical = np.sqrt((k - 1) * stats.f.ppf(1 - alpha, k - 1, df_error)) * se
            pairs.append({"group1": name_a, "group2": name_b, "mean_difference": difference,
                          "standard_error": se, "p_value": p, "lower": difference - critical,
                          "upper": difference + critical, "reject": bool(abs(difference) > critical)})
    if method == "bonferroni":
        rejected, adjusted, _, _ = multipletests(raw, alpha=alpha, method="bonferroni")
        for row, p, reject in zip(pairs, adjusted, rejected): row.update({"p_value": p, "reject": bool(reject)})
    return json_safe({"method": method, "comparisons": pairs, "mse": mse, "degrees_of_freedom_error": df_error})


def repeated(data: pd.DataFrame) -> dict:
    result = AnovaRM(data, depvar="value", subject="subject", within=["within"], aggregate_func=None).fit()
    return {"table": _table(result.anova_table), "subjects": int(data["subject"].nunique()),
            "levels": json_safe(data["within"].drop_duplicates().tolist())}
