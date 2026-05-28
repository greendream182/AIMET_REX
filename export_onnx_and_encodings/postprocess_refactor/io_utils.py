from __future__ import annotations

from typing import Any, Dict


def load_encodings(encodings_path: str) -> Dict[str, Any]:
    try:
        import json

        with open(encodings_path, "r") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法读取 encodings: {exc}") from exc


def save_encodings(enc: Dict[str, Any], encodings_path: str) -> None:
    import json

    with open(encodings_path, "w") as f:
        json.dump(enc, f, indent=2)
