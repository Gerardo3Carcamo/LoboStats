from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

import pandas as pd

from utils.errors import APIError
from utils.serialization import dataframe_records


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    frame: pd.DataFrame
    created_at: str

    def metadata(self) -> dict:
        columns = []
        for column in self.frame.columns:
            series = self.frame[column]
            if pd.api.types.is_numeric_dtype(series):
                data_type = "numérico"
            elif pd.api.types.is_datetime64_any_dtype(series):
                data_type = "fecha"
            elif pd.api.types.is_bool_dtype(series):
                data_type = "booleano"
            else:
                data_type = "categórico"
            columns.append({
                "name": str(column),
                "type": data_type,
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            })
        return {
            "id": self.id,
            "dataset_id": self.id,
            "name": self.name,
            "columns": columns,
            "row_count": len(self.frame),
            "column_count": len(self.frame.columns),
            "created_at": self.created_at,
            "preview": dataframe_records(self.frame.head(10)),
        }


class DatasetStore:
    """Repositorio volátil para el MVP; los datos se pierden al reiniciar."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._lock = RLock()

    def create(self, name: str, frame: pd.DataFrame) -> dict:
        if frame.columns.empty:
            raise APIError("El dataset debe contener al menos una columna.")
        normalized = frame.copy()
        normalized.columns = [str(column) for column in normalized.columns]
        if len(set(normalized.columns)) != len(normalized.columns):
            raise APIError("Los nombres de columnas deben ser únicos.")
        dataset = Dataset(
            id=str(uuid4()),
            name=name,
            frame=normalized,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._datasets[dataset.id] = dataset
        return dataset.metadata()

    def get(self, dataset_id: str) -> Dataset:
        if not isinstance(dataset_id, str):
            raise APIError("'dataset_id' debe ser un texto.", details={"field": "dataset_id"})
        with self._lock:
            dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise APIError("Dataset no encontrado.", 404, "dataset_not_found", {"dataset_id": dataset_id})
        return dataset

    def delete(self, dataset_id: str) -> None:
        if not isinstance(dataset_id, str):
            raise APIError("'dataset_id' debe ser un texto.", details={"field": "dataset_id"})
        with self._lock:
            if self._datasets.pop(dataset_id, None) is None:
                raise APIError("Dataset no encontrado.", 404, "dataset_not_found", {"dataset_id": dataset_id})
