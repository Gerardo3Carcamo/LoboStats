import numpy as np
import pandas as pd
from scipy import stats

from modules.dataset_store import Dataset
from utils.serialization import dataframe_records, json_safe


def histogram_data(values: pd.Series, bins: int, density: bool = False, kde: bool = False) -> dict:
    counts, edges = np.histogram(values, bins=bins, density=density)
    result = {"counts": counts, "bin_edges": edges, "sample_size": len(values), "density": density, "kde": []}
    if kde and len(values) >= 2 and values.nunique() > 1:
        x = np.linspace(values.min(), values.max(), min(200, max(50, bins * 4)))
        result["kde"] = [{"x": a, "y": b} for a, b in zip(x, stats.gaussian_kde(values)(x))]
    return json_safe(result)


def _box_stats(values: pd.Series) -> dict:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    return json_safe({
        "count": len(values),
        "min": values.min(),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": values.max(),
        "lower_whisker": inside.min(),
        "upper_whisker": inside.max(),
        "outliers": values[(values < lower_fence) | (values > upper_fence)].tolist(),
    })


def boxplot_data(dataset: Dataset, column: str, group_by: str | None) -> dict:
    if group_by is None:
        values = pd.to_numeric(dataset.frame[column], errors="coerce").dropna().astype(float)
        return {"groups": [{"group": None, **_box_stats(values)}]}
    groups = []
    for group, frame in dataset.frame.groupby(group_by, dropna=False):
        values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)
        if not values.empty:
            groups.append({"group": json_safe(group), **_box_stats(values)})
    return {"groups": groups}


def scatter_data(dataset: Dataset, x: str, y: str) -> dict:
    pairs = dataset.frame[[x, y]].copy()
    pairs[x] = pd.to_numeric(pairs[x], errors="coerce")
    pairs[y] = pd.to_numeric(pairs[y], errors="coerce")
    pairs = pairs.dropna()
    points = [{"x": float(a), "y": float(b)} for a, b in pairs.itertuples(index=False, name=None)]
    result = {"points": points, "regression_line": [], "count": len(pairs)}
    if len(pairs) >= 2 and pairs[x].nunique() > 1:
        slope, intercept = np.polyfit(pairs[x], pairs[y], 1)
        x_min, x_max = float(pairs[x].min()), float(pairs[x].max())
        predicted = slope * pairs[x] + intercept
        residual_sum = float(((pairs[y] - predicted) ** 2).sum())
        total_sum = float(((pairs[y] - pairs[y].mean()) ** 2).sum())
        result.update({
            "regression_line": [
                {"x": x_min, "y": float(slope * x_min + intercept)},
                {"x": x_max, "y": float(slope * x_max + intercept)},
            ],
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": 1 - residual_sum / total_sum if total_sum else None,
        })
    return result


def sampled_frame(frame: pd.DataFrame, columns: list[str], limit: int) -> tuple[pd.DataFrame, bool]:
    data = frame[columns].dropna()
    sampled = len(data) > limit
    if sampled:
        data = data.iloc[np.linspace(0, len(data) - 1, limit, dtype=int)]
    return data, sampled


def grouped_chart(frame: pd.DataFrame, category: str, value: str | None, aggregation: str) -> list[dict]:
    data = frame[[category] + ([value] if value else [])].dropna()
    if value:
        data[value] = pd.to_numeric(data[value], errors="coerce")
        grouped = data.dropna().groupby(category, dropna=False)[value].agg(aggregation).sort_values(ascending=False)
    else:
        grouped = data[category].value_counts(dropna=False)
    return [{"category": json_safe(name), "value": float(amount)} for name, amount in grouped.iloc[:100].items()]


def numeric_points(frame: pd.DataFrame, columns: list[str], limit: int = 5000) -> dict:
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    data, sampled = sampled_frame(data, columns, limit)
    return {"points": dataframe_records(data), "count": len(data), "sampled": sampled, "max_points": limit}
