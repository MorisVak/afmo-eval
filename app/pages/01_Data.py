"""Utilities for AFMo (app/pages/01_Data.py)"""
import _bootstrap  # ensure src/ is on sys.path if not installed
import streamlit as st
import numpy as np
import pandas as pd
import html
from typing import Optional
from utils.state_utils import ensure_state, get_current_dataframe
from afmo.io import load_dataframe_multi, to_csv_bytes
from afmo.plot import history_plot
from afmo.features import get_feature_values
from utils.theme_utils import theme_header
ensure_state()
theme_header("Data", key="hdr_import")
st.session_state["_active_page"] = "Data"

st.markdown("""
<style>
/* Small circular help icon */
.help-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  margin-left: 6px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.6);
  color: rgba(255,255,255,0.8);
  font-size: 10px;
  font-weight: 600;
  line-height: 1;
  cursor: help;
}

/* Tooltip container */
.help-tooltip {
  position: relative;
  display: inline-block;
}

/* Tooltip box */
.help-tooltip .tooltip-text {
  visibility: hidden;
  width: 320px;
  background-color: #262730;
  color: #fff;
  text-align: left;
  border-radius: 6px;
  padding: 10px 12px;
  position: absolute;
  z-index: 1000;
  top: 130%;
  left: 0;
  box-shadow: 0 4px 14px rgba(0,0,0,0.4);
  font-size: 0.9rem;
  line-height: 1.4;
}

/* Show tooltip on hover */
.help-tooltip:hover .tooltip-text {
  visibility: visible;
}
</style>
""", unsafe_allow_html=True)

# Quick actions for AFMo/examples
from pathlib import Path as _Path
_examples_dir = _Path(__file__).resolve().parents[2] / "examples"

# Define infer_freq checkbox early so it can be used by both loaders
infer_freq = st.checkbox("Infer frequency automatically", value=True, key="data_infer_freq")

with st.expander("📁 Quick access: AFMo/examples"):
    # Load from examples
    try:
        example_files = sorted(
            [p for p in _examples_dir.glob("*") if p.suffix.lower() in {".csv", ".json", ".parquet"}])
    except Exception:
        example_files = []
    chosen = st.selectbox("Load example data", options=["-- select --"] + [p.name for p in example_files], key="ex_sel_data")
    colx, coly = st.columns([1,1])
    with colx:
        if st.button("Load selected example"):
            if chosen and chosen != "-- select --":
                p = _examples_dir / chosen
                try:
                    with p.open("rb") as fh:
                        df = load_dataframe_multi(fh)
                    st.session_state.data = df
                    if df is not None and hasattr(df, "columns") and len(df.columns):
                        st.session_state.active_col = df.columns[0]
                    # Infer frequency for quick-loaded data
                    if infer_freq and df is not None and getattr(df, "index", None) is not None:
                        try:
                            freq = pd.infer_freq(df.index)
                        except Exception:
                            freq = 1
                        st.session_state.freq = freq if freq else 1
                    st.success(f"Loaded example: {p.name}")
                except Exception as e:
                    st.error(f"Failed to load example: {e}")
    with coly:
        # Save current data to examples
        if st.session_state.get("data") is not None:
            if st.button("Save current data to examples as CSV"):
                try:
                    outp = _examples_dir / "afmo_data_export.csv"
                    st.session_state["data"].to_csv(outp)  # write DataFrame to CSV
                    st.success(f"Saved to {outp}")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        else:
            st.caption("Load data first to enable saving.")


# Data page: Test Mode is disabled
st.session_state.setdefault("test_mode", False)
st.session_state.setdefault("data_train", None)
st.session_state.setdefault("data_test", None)
st.session_state.setdefault("test_windows", [])


st.subheader("Load data")
left, right = st.columns([1, 2], gap="large")

with left:
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])

    # Toggle + horizon (h) are ALWAYS visible
    tm_col1, tm_col2 = st.columns([1, 1])
    # Test Mode UI removed on Data page per requirements.

    # Load data (if provided)
    if file is not None:
        try:
            df = load_dataframe_multi(file)
            # Fallback: if df is empty, try robust on-page parser with multiple variants
            if df is None or getattr(df, "empty", True):
                import pandas as pd, io as _io
                raw = getattr(file, "getvalue", lambda: None)()
                if isinstance(raw, (bytes, bytearray)):
                    bio = _io.BytesIO(raw)
                else:
                    # last resort: read() then rewind
                    try:
                        pos = file.tell()
                        raw = file.read()
                        file.seek(pos)
                        bio = _io.BytesIO(raw)
                    except Exception:
                        bio = None
                def _try_parsers(bio_bytes):
                    if bio_bytes is None:
                        return None
                    data = bio_bytes.getvalue()
                    # Detect Excel magic
                    is_excel = str(getattr(file, "name", "")).lower().endswith((".xlsx",".xls")) or (data[:2] == b'PK')
                    if is_excel:
                        for engine in (None, "openpyxl", "xlrd"):
                            try:
                                return pd.read_excel(_io.BytesIO(data), engine=engine if engine else None)
                            except Exception:
                                continue
                    # CSV trials
                    seps = [",",";","\t", None]
                    decs = [".", ","]
                    encs = ["utf-8","latin-1","cp1252"]
                    headers = ["infer", None]
                    idxs = [0, None]
                    for enc in encs:
                        for sep in seps:
                            for dec in decs:
                                for header in headers:
                                    for idx in idxs:
                                        try:
                                            df2 = pd.read_csv(_io.BytesIO(data), encoding=enc, sep=sep, decimal=dec,  # read CSV file
                                                              header=None if header is None else "infer",
                                                              index_col=idx if header is not None else None, engine="python")
                                            if isinstance(df2, pd.DataFrame) and df2.size > 0:
                                                # Try to promote first column to datetime index if it looks like dates
                                                try:
                                                    if df2.shape[1] >= 2:
                                                        pd.to_datetime(df2.iloc[:,0], errors="raise")  # position-based indexing into DataFrame
                                                        df2 = df2.set_index(pd.to_datetime(df2.iloc[:,0], errors="coerce")).iloc[:,1:]  # position-based indexing into DataFrame
                                                except Exception:
                                                    pass
                                                return df2
                                        except Exception:
                                            continue
                    return None
                alt = _try_parsers(bio)
                if alt is not None and not alt.empty:
                    df = alt
            st.session_state.data = df
            if df is not None and hasattr(df, "columns") and len(df.columns.tolist())>0:
                st.session_state.active_col = df.columns[0]
            if infer_freq and df is not None and getattr(df, "index", None) is not None:
                try:
                    freq = pd.infer_freq(df.index)
                except Exception:
                    freq = 1
                if not freq:
                    st.session_state.freq = 1
                else:
                    st.session_state.freq = freq
            if df is not None and not df.empty:
                st.caption(
                    f"✅ Loaded data: {len(df)} rows × {len(df.columns)} series. "
                    f"Start: {getattr(df.index, 'min', lambda: None)()} | End: {getattr(df.index, 'max', lambda: None)()}"
                )
            else:
                st.warning("No rows to show — check delimiter/decimals/date parsing. Try using a different delimiter or decimal, or provide a sample with a date column plus one or more value columns.")
        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")

    else:
        st.caption("Choose a file to load (CSV, XLSX, XLS). First column must be dates; others numeric.")
        
with right:
    df_full = st.session_state.data
    df_use = get_current_dataframe()

    if df_full is None:
        st.info("Please upload a CSV. Format: first column = datetime, remaining columns = time series.")
    else:
        st.subheader("Preview")
        _freq = st.session_state.get("freq")

        if isinstance(df_use, pd.DataFrame) and not df_use.empty:
            # Dataset summary
            with st.container(border=True):
                st.markdown("#### Dataset summary")
                rows = len(df_use)
                columns = len(df_use.columns)
                start = str(df_use.index.min()).split(" ", 1)[0]
                end = str(df_use.index.max()).split(" ", 1)[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Rows", f"{rows:,}")
                c2.metric("Collumns", f"{columns}")
                c3.metric("Start", f"{start}")
                c4.metric("End", f"{end}")
                c5.metric("Frequency", _freq if _freq else "—")

        else:
            st.warning("No rows to show — check delimiter/decimals/date parsing.")
            # Still show an empty summary so layout is consistent
            with st.container(border=True):
                st.markdown("#### Dataset summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", "0")
                c2.metric("Start", "—")
                c3.metric("End", "—")
                c4.metric("Frequency", _freq if _freq else "—")

    st.markdown("---")
    st.subheader("Series selection, plot & diagnostics")


if get_current_dataframe() is None:
    st.caption("Load data to select series and view diagnostics.")
else:
    df = get_current_dataframe()
    # Sanitize previously selected series (remove non-existing)
    prev_sel = st.session_state.get('selected_cols', [])
    if isinstance(prev_sel, list) and prev_sel:
        prev_sel = [c for c in prev_sel if c in df.columns]
        st.session_state['selected_cols'] = prev_sel
    default_sel = prev_sel if prev_sel else ([st.session_state.active_col] if (st.session_state.active_col in df.columns) else [df.columns[0]])
    selected_cols = st.multiselect(
        "Select one or more series for plotting",
        options=list(df.columns),
        default=default_sel,
        key="data_selected_series"
    )
    st.session_state.selected_cols = selected_cols
    if selected_cols:
        st.session_state.active_col = selected_cols[0]
        st.plotly_chart(history_plot(df, selected_cols), use_container_width=True)
    else:
        st.info("Please select at least one series to plot and diagnose.")

    st.markdown("---")
    st.subheader("Statistical diagnostics")
    def _interpret(metric: str, value: Optional[float]) -> str:
        try:
            from afmo.core.registry import FEATURES
            func = FEATURES.get(metric)
            interp = getattr(func, 'interpret', None) if func is not None else None
            if callable(interp):
                return interp(value)
        except Exception:
            pass
        if value is None:
            return "No result (insufficient data)"
        return "Result available"

    from afmo.core.registry import FEATURES as _FEATREG
    notes = {name: getattr(func, "note", "") for name, func in _FEATREG.items()}


#changed to where diagnostics of every series is visble by default. No need to select.
    def format_value(val, decimals=4, max_len=14):
        if val is None:
            return "—"
        s = f"{val:.{decimals}f}"
        if len(s) > max_len:
            s = f"{val:.{decimals}e}"
        return s
    
    ##added css such that overflown numbers are still shown properly
    st.markdown("""
    <style>
    div[data-testid="stExpander"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    }
    </style>
    """, unsafe_allow_html=True)

    rows_all = []
    with st.spinner("Computing tests…"):
        freq = st.session_state.get("freq")
        for col in df.columns:
            y = df[col]
            results = get_feature_values(y, freq)
            for metric, val in results.items():
                val_f = None if val is None else float(val)
                disp_str = format_value(val_f, decimals=4, max_len=14)
                full_str = "—" if val_f is None else f"{val_f:.16g}"

                rows_all.append(
                    {
                        "Series": col,
                        "Metric": metric,
                        "Value": disp_str,        
                        "Value_full": full_str,     
                        "Note": notes.get(metric, ""),
                        "Interpretation": _interpret(metric, val),
                    }
                )

    out = pd.DataFrame(rows_all)[["Series", "Metric", "Value", "Value_full", "Note","Interpretation"]]

    selected_cols = st.multiselect(
        "Select one or more series for diagnostics",
        options=list(df.columns),
        key="data_selected_diagnostics"
    )

    # Filter diagnostics to selected series
    out_view = out if not selected_cols else out[out["Series"].isin(selected_cols)]

    groups = list(out_view.groupby("Series"))
    chunk_size = 3

    expand_all_diagnostics = st.toggle("Expand all series", value=False, key="expand_all_diagnostics")

    for i in range(0, len(groups), chunk_size):
        row_groups = groups[i:i + chunk_size]
        cols = st.columns(len(row_groups))

        for (series_name, group), col in zip(row_groups, cols):
            with col:
                with st.expander(series_name, expanded=expand_all_diagnostics):
                    for _, row in group.iterrows():
                        with st.container():
                            metric = html.escape(str(row["Metric"]))
                            note = html.escape((row.get("Note") or "").strip())

                            # Metric title + tooltip icon
                            if note:
                                st.markdown(
                                    f"""
                                    <h3 style="margin-bottom: 0.25rem; display: flex; align-items: center;">
                                    <span>{metric}</span>
                                    <span class="help-tooltip">
                                        <span class="help-icon">?</span>
                                        <span class="tooltip-text">
                                        {note}
                                        </span>
                                    </span>
                                    </h3>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(f"### {metric}")

                            metric_cols = st.columns([1, 3])

                            with metric_cols[0]:
                                st.markdown(
                                    f"""
                                    <div style="display:flex; flex-direction:column; gap:0.15rem;">
                                    <div style="font-weight:600;">Value</div>
                                    <div style="font-size:1.6rem; font-weight:700; line-height:1.1;">
                                        {row['Value']}
                                    </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with metric_cols[1]:
                                st.markdown(
                                    f"""
                                    <div style="display:flex; flex-direction:column; gap:0.15rem; padding-left: 2px;">
                                    <div style="font-weight:600;">Interpretation</div>
                                    <div style="font-size:0.95rem; line-height:2.0; opacity:0.85;">
                                        {html.escape(str(row["Interpretation"]))}
                                    </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            st.markdown("---")

