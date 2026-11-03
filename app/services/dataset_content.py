import hashlib
import json
from typing import Any


def dataset_content_sha256(columns: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
