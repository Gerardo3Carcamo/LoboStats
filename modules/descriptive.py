import numpy as np
import pandas as pd
from scipy import stats

from modules.dataset_store import Dataset
from utils.serialization import json_safe


NUMERIC_FIELDS = (
    "mean", "median", "std", "variance", "min", "q1", "q3", "max",
    "range", "iqr", "cv", "sem", "skewness", "kurtosis",
)


def summarize(dataset: Dataset, columns: list[str], alpha: float) -> dict:
    summaries = {}
    for column in columns:
        source = dataset.frame[column]
        non_null = source.dropna()
        modes = non_null.mode().tolist()
        item = {
            "count": int(non_null.count()),
            "missing": int(source.isna().sum()),
            "mode": json_safe(modes[0]) if modes else None,
            **{field: None for field in NUMERIC_FIELDS},
            "shapiro": None,
        }
        numeric = pd.to_numeric(non_null, errors="coerce").dropna().astype(float)
        if not numeric.empty:
            count = len(numeric)
            mean = numeric.mean()
            q1, q3 = numeric.quantile([0.25, 0.75])
            std = numeric.std(ddof=1) if count > 1 else np.nan
            item.update({
                "mean": mean,
                "median": numeric.median(),
                "std": std,
                "variance": numeric.var(ddof=1) if count > 1 else np.nan,
                "min": numeric.min(),
                "q1": q1,
                "q3": q3,
                "max": numeric.max(),
                "range": numeric.max() - numeric.min(),
                "iqr": q3 - q1,
                "cv": (std / mean * 100) if count > 1 and mean != 0 else np.nan,
                "sem": stats.sem(numeric) if count > 1 else np.nan,
                "skewness": stats.skew(numeric, bias=False) if count > 2 and numeric.nunique() > 1 else np.nan,
                "kurtosis": stats.kurtosis(numeric, bias=False) if count > 3 and numeric.nunique() > 1 else np.nan,
            })
            if 3 <= count <= 5000 and numeric.nunique() > 1:
                statistic, p_value = stats.shapiro(numeric)
                item["shapiro"] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "is_normal": bool(p_value >= alpha),
                    "interpretation": (
                        "No se rechaza la normalidad de los datos."
                        if p_value >= alpha
                        else "Se rechaza la normalidad de los datos."
                    ),
                }
        summaries[column] = json_safe(item)
    return summaries


def frequencies(dataset: Dataset, column: str) -> list[dict]:
    series = dataset.frame[column]
    total = len(series)
    counts = series.value_counts(dropna=False)
    return [
        {
            "value": json_safe(value),
            "count": int(count),
            "percentage": float(count / total * 100) if total else 0.0,
        }
        for value, count in counts.items()
    ]
