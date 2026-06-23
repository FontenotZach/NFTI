from __future__ import annotations

import math
from typing import Any


# Canonical missing sentinel for TraumaRecord field values.
MISSING = float("nan")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def field_value_from_row(data_row, header_name: str) -> Any:
    """
    Read a header value from a data row. Missing columns use MISSING (NaN);
    present values (including valid 0) are preserved.
    """
    if header_name not in data_row:
        return MISSING
    return data_row[header_name]


def biu_indicates_missing(biu_value: Any) -> bool:
    """Return True when a BIU code marks the sister field as not observed."""
    if is_missing(biu_value):
        return False
    try:
        code = int(biu_value)
    except (TypeError, ValueError):
        return False
    # NTDS-style BIU: 1 = Not Applicable, 2 = Not Known / Not Recorded
    return code in (1, 2)
