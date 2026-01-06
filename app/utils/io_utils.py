
"""Shim for legacy imports: utils.io_utils
Provides load_dataframe_multi(...) and to_csv_bytes(...)
"""
from __future__ import annotations
import io
from typing import Iterable, Union, List, Any
import pandas as pd
import streamlit as st

FileLike = Union[str, bytes, io.BytesIO, io.BufferedReader, Any]

# selected series list from Data page (fallback to single selection)
def _get_selected_series_list(df: pd.DataFrame, fallback: str) -> list[str]:
    sel = st.session_state.get('selected_cols', [])
    if isinstance(sel, list) and len(sel) > 0:
        return [c for c in sel if c in df.columns]
    return [fallback]

def infer_m_from_freq(freq, y_index=None, default=0):
    """Map pandas freq alias ."""
    if freq is None and y_index is not None:
        try:
            freq = pd.infer_freq(y_index)
        except Exception:
            freq = None

    if not freq:
        return default

    f = str(freq).upper()
    # common aliases
    if f in {"A", "AS", "Y", "YS"}:   return 1
    if f in {"Q", "QS"}:              return 4
    if f in {"M", "MS"}:              return 12
    if f.startswith("W"):             return 52           # "W", "W-MON", ...
    if f in {"D"}:                    return 7            # weekly
    if f in {"B"}:                    return 5            # weekday
    if f in {"H"}:                    return 24           # dayly
    if f in {"T", "MIN"}:             return 60           # hourly
    if f in {"S"}:                    return 60           # minute
    return default

def _read_any(f: FileLike) -> pd.DataFrame:
    """Read CSV or Excel from various inputs. Tries sensible defaults (UTF-8, ; or , delimiter)."""
    if hasattr(f, "name"):
        name = getattr(f, "name")
    elif isinstance(f, (str, bytes)):
        name = f if isinstance(f, str) else "uploaded.bin"
    else:
        name = getattr(f, "name", "uploaded.bin")
    lower = str(name).lower()

    # Build a BytesIO buffer for UploadedFile-like objects
    if hasattr(f, "getvalue"):
        data = f.getvalue()
        bio = io.BytesIO(data if isinstance(data, (bytes, bytearray)) else bytes(data))
    elif isinstance(f, (bytes, bytearray)):
        bio = io.BytesIO(f)
    elif isinstance(f, str):
        # path
        if lower.endswith(('.xlsx', '.xls')):
            return pd.read_excel(f)
        else:
            # try csv
            try:
                return pd.read_csv(f)
            except Exception:
                return pd.read_csv(f, sep=';')
    else:
        # file-like
        try:
            pos = f.tell()
        except Exception:
            pos = None
        try:
            data = f.read()
        finally:
            try:
                if pos is not None:
                    f.seek(pos)
            except Exception:
                pass
        bio = io.BytesIO(data if isinstance(data, (bytes, bytearray)) else bytes(data))

    # Decide by extension
    if lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(bio)
    # Try CSV UTF-8, then semicolon
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio, sep=';')

def load_dataframe_multi(files: Union[FileLike, Iterable[FileLike]]) -> pd.DataFrame:
    """Load one or multiple files and concatenate vertically (add 'source' if multiple).
    - Accepts Streamlit UploadedFile, file paths, bytes, or file-like objects.
    - Parses datetime index if column named 'date' exists.
    """
    if files is None:
        raise ValueError("No files provided")
    if not isinstance(files, (list, tuple)):
        files = [files]

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = _read_any(f)
        # Add source if available
        src = getattr(f, 'name', None) if hasattr(f, 'name') else (f if isinstance(f, str) else None)
        if src is not None:
            df = df.copy()
            if 'source' not in df.columns:
                df['source'] = src
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=False)
    # Try to set datetime index from 'date' or first column
    if 'date' in out.columns:
        try:
            out['date'] = pd.to_datetime(out['date'])
            out = out.set_index('date').sort_index()
        except Exception:
            pass
    elif out.index.name and 'date' in str(out.index.name).lower():
        try:
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
        except Exception:
            pass
    return out

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return UTF-8 encoded CSV bytes for download buttons."""
    return df.to_csv(index=True).encode('utf-8')
