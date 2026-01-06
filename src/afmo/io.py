"""Utilities for AFMo (src/afmo/io.py)"""
import io
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

def load_dataframe_multi(file) -> pd.DataFrame:
    """
    Robust loader for tabular time series data.

    Behavior change:
      - First column is parsed to datetime ONLY if it truly looks like dates.
        If the column is numeric, it stays numeric.
    """
    import os
    from pandas.api.types import is_numeric_dtype

    # Detect filename to decide Excel vs CSV
    name = getattr(file, "name", "")
    is_excel = isinstance(name, str) and name.lower().endswith((".xlsx", ".xls"))

    def _as_io(f):
        # Ensure we can seek (Streamlit uploads are file-like)
        try:
            f.seek(0)
            return f
        except Exception:
            return io.BytesIO(f.read())

    def _try_read_csv(f):
        f = _as_io(f)
        try:
            df = pd.read_csv(f, sep=None, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
        seps = [",", ";", "\t", "|"]
        decimals = [".", ","]
        for sep in seps:
            for dec in decimals:
                f = _as_io(file)
                try:
                    df = pd.read_csv(f, sep=sep, decimal=dec)
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    continue
        f = _as_io(file)
        return pd.read_csv(f)

    if is_excel:
        f = _as_io(file)
        df = pd.read_excel(f)
    else:
        df = _try_read_csv(file)

    if df.shape[1] < 2:
        raise ValueError(
            "Data must have at least two columns: an index-like column + one or more series columns."
        )

    date_col = df.columns[0]
    value_cols = list(df.columns[1:])

    # Decide if first column is a date-like column
    col0 = df[date_col]

    def _looks_like_dates(series: pd.Series, sample=500, min_ok_ratio=0.8, min_non_na=5, min_unique=3) -> bool:
        # Only try to parse strings/objects; numeric stays numeric by default
        if is_numeric_dtype(series):
            return False
        # sample to keep it cheap
        s = series.dropna().astype(str)
        if s.empty:
            return False
        s = s.head(sample)
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        ok_ratio = parsed.notna().mean()
        # also require enough non-NaTs and some variability
        return (ok_ratio >= min_ok_ratio) and (parsed.notna().sum() >= min_non_na) and (parsed.nunique() >= min_unique)

    is_date = _looks_like_dates(col0)

    if is_date:
        # Parse datetime and drop rows where parsing failed
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=[date_col])
    else:
        # Keep numeric (or string) as-is; if it's numeric-like strings, coerce to numeric
        if not is_numeric_dtype(df[date_col]):
            maybe_num = pd.to_numeric(df[date_col], errors="ignore")
            df[date_col] = maybe_num

    # Coerce numeric series columns
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Set index & sort (works for datetime, numeric, or string)
    df = df.set_index(date_col).sort_index()

    # Drop completely empty value columns
    keep = [c for c in value_cols if df[c].notna().any()]
    if not keep:
        raise ValueError(
            "No valid numeric series columns found after parsing. "
            "Check separators and decimals (we tried ',', ';', tab, '|' with '.' and ',')."
        )
    df = df[keep]

    # Remove duplicate index entries by keeping the last occurrence
    df = df[~df.index.duplicated(keep="last")]

    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf)  # write DataFrame to CSV
    return buf.getvalue().encode("utf-8")

def export_session(data: Optional[pd.DataFrame], meta: Dict[str, Any], forecasts: Dict[str, pd.DataFrame]) -> bytes:
    payload: Dict[str, Any] = {"meta": meta, "data": None, "forecasts": {}}
    if data is not None:
        payload["data"] = {
            "index": [str(x) for x in data.index],
            "columns": list(data.columns),
            "values2d": data.reset_index(drop=True).values.tolist(),
        }
    for k, v in forecasts.items():
        payload["forecasts"][k] = {
            "index": [str(x) for x in v.index],
            "columns": list(v.columns),
            "values": v.reset_index(drop=True).values.tolist(),
        }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")  # serialize to JSON text

def import_session(file) -> Tuple[Optional[pd.DataFrame], dict, dict]:
    obj = json.loads(file.read())  # parse JSON text
    data_df = None
    if obj.get("data"):
        d = obj["data"]
        idx = pd.to_datetime(d.get("index", []), errors="coerce")
        cols = d.get("columns", [])
        if "values2d" in d:
            arr = np.array(d["values2d"]) if len(d["values2d"]) else np.empty((0, len(cols)))
            data_df = pd.DataFrame(arr, columns=cols, index=idx)
        elif "values" in d:
            s = pd.Series(d["values"], index=idx, name=cols[0] if cols else "y")
            data_df = s.to_frame()
    meta = obj.get("meta", {})
    forecasts = {}
    for k, d in obj.get("forecasts", {}).items():
        idx = pd.to_datetime(d.get("index", []), errors="coerce")
        cols = d.get("columns", [])
        vals = d.get("values", [])
        arr = np.array(vals)
        forecasts[k] = pd.DataFrame(arr, columns=cols, index=idx)
    return data_df, meta, forecasts