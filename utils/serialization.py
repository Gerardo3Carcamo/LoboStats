import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Convierte tipos científicos y valores no finitos a JSON estricto."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    return [json_safe(record) for record in frame.to_dict(orient="records")]
