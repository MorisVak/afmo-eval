import streamlit as st
from typing import Optional, Callable

def _css_dark_soften() -> str:
    return """
    <style>
    .stButton > button { border-radius: 8px !important; }
    </style>
    """

def apply_theme(theme_name: str) -> None:
    st.markdown(_css_dark_soften(), unsafe_allow_html=True)

def theme_header(title: str, key: str, right_extra: Optional[Callable[[], None]] = None) -> None:
    """Page header with Theme toggle and Session controls, visible on every page."""
    # Safe defaults
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("use_app_theme", True)

    col1, col2, col3 = st.columns([1, 0.25, 0.35])
    with col3:
        session_controls(compact=True, key_prefix=key)
 
        # Custom right-side extra (page-supplied)
        if right_extra is not None:
            try:
                right_extra()
            except Exception:
                pass

    # Apply app CSS only if explicitly enabled
    if st.session_state.get("use_app_theme", True):
        apply_theme(st.session_state["theme"])

def _json_safe(obj):
    """ _json_safe"""
    import numpy as _np, pandas as _pd, datetime as _dt, json as _json
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (_dt.datetime,)):
        return obj.isoformat()
    if isinstance(obj, (_dt.date,)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, (dict,)):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        _json.dumps(obj)  # serialize to JSON text
        return obj
    except Exception:
        return str(obj)

def _build_session_bytes() -> bytes:
    """ _build_session_bytes"""
    import pandas as pd
    from afmo.io import export_session as _export

    ss = st.session_state

    # Data
    data = ss.get("data", None)
    data_df = data.copy() if isinstance(data, pd.DataFrame) else None

    # Forecasts (ins + oos)
    forecasts = {}
    for k, v in ss.get("forecasts", {}).items():
        forecasts[str(k)] = v
    for k, v in ss.get("oos_forecasts", {}).items():
        forecasts[f"oos:{k}"] = v

    # Meta
    meta = {
        "freq": ss.get("freq", None),
        "active_col": ss.get("active_col", None),
        "h": int(ss.get("h", 1)) if ss.get("h", None) is not None else None,
    }

    # UI theme
    meta["ui"] = {"theme": ss.get("theme", None), "use_app_theme": ss.get("use_app_theme", True)}

    # Evaluation & controls (broad capture)
    import re as _re
    capture = {}
    for k in list(ss.keys()):
        if _re.match(r"^(eval_|summary_|val_|bt_|analysis_)", str(k)) or k in {"test_windows", "fc_horizon"}:
            capture[k] = _json_safe(ss.get(k, None))
    meta["evaluation"] = capture

    # Universal lightweight session snapshot (exclude heavy objects handled separately)
    blacklist = {"data", "models", "forecasts", "oos_forecasts"}
    ss_dump = {}
    for k in list(ss.keys()):
        if k in blacklist:
            continue
        # Skip problematic UI/widget keys
        ks = str(k)
        if ks.startswith("hdr_") or ks.startswith("__") or ks.startswith("ex_sel_"):
            continue
        # Drop known button/run keys to avoid StreamlitAPIException on load
        if ks == "analysis_run" or ks.endswith("_run") or ks.startswith("btn_") or "button" in ks:
            continue
        try:
            ss_dump[str(k)] = _json_safe(ss.get(k))
        except Exception:
            pass
    meta["ss"] = ss_dump

    # Explicit selections
    selections = {}
    for k in ("selected_cols", "data_selected_series", "forecast_series_multiselect", "eval_selected_models"):
        if k in ss:
            selections[k] = _json_safe(ss.get(k))
    meta["selections"] = selections

    return _export(data_df, meta, forecasts)

def _apply_loaded_session(file) -> str:
    import pandas as pd
    from afmo.io import import_session as _import

    df, meta, fcs = _import(file)

    # Data
    st.session_state["data"] = df

    # Meta
    meta = meta or {}
    st.session_state["freq"] = meta.get("freq", None)
    st.session_state["active_col"] = meta.get("active_col", None)

    # Horizon (robust)
    raw_h = meta.get("h", st.session_state.get("h", 1))
    try:
        h = int(raw_h) if raw_h is not None else 1
    except (ValueError, TypeError):
        h = 1
    st.session_state["h"] = h

    # Models reset (not serialized)
    st.session_state["models"] = {}

    # Forecasts
    ins, oos = {}, {}
    if isinstance(fcs, dict) and fcs:
        for k, v in fcs.items():
            k = str(k)
            if k.startswith("oos:"):
                oos[k.split("oos:", 1)[1]] = v
            else:
                ins[k] = v
    st.session_state["forecasts"] = ins
    st.session_state["oos_forecasts"] = oos

    # Lightweight session snapshot
    ss_dump = meta.get("ss", {}) if isinstance(meta, dict) else {}
    if isinstance(ss_dump, dict) and ss_dump:
        for k, v in ss_dump.items():
            if k in {"data", "models", "forecasts", "oos_forecasts"}:
                continue
            # Skip UI-only ephemeral keys that break Streamlit on restore
            if str(k).startswith(("hdr_", "__", "ex_sel_", "btn_")) or str(k)=="analysis_run" or str(k).endswith("_run") or "button" in str(k):
                continue
            st.session_state[k] = v

    # Evaluation
    eval_meta = meta.get("evaluation", {}) if isinstance(meta, dict) else {}
    if isinstance(eval_meta, dict) and eval_meta:
        for k, v in eval_meta.items():
            st.session_state[k] = v

    # Selections
    sel_meta = meta.get("selections", {}) if isinstance(meta, dict) else {}
    if isinstance(sel_meta, dict) and sel_meta:
        for k, v in sel_meta.items():
            st.session_state[k] = v

    # UI theme
    ui_meta = meta.get("ui", {}) if isinstance(meta, dict) else {}
    if isinstance(ui_meta, dict):
        if ui_meta.get("theme") is not None:
            st.session_state["theme"] = ui_meta.get("theme")
        if "use_app_theme" in ui_meta:
            st.session_state["use_app_theme"] = bool(ui_meta.get("use_app_theme"))
        if st.session_state.get("use_app_theme", True) and st.session_state.get("theme") is not None:
            try:
                apply_theme(st.session_state["theme"])
            except Exception:
                pass

    return "Session loaded. Data, settings, forecasts, evaluation results, selections, UI theme, and UI state restored. Models were not saved and must be refit if needed."


# Header (Theme + Session)

def session_controls(compact: bool = True, key_prefix: str = "") -> None:
    """ session_controls"""
    if hasattr(st, "popover") and compact:
        with st.popover("🗂 Session"):
            st.caption("Save or load the current session.")
            _render_session_buttons(key_prefix=key_prefix)
    else:
        st.caption("Session")
        _render_session_buttons(key_prefix=key_prefix)

def _render_session_buttons(key_prefix: str = "") -> None:
    """_render_session_buttons"""
    col_a, col_b = st.columns([1, 1])
    with col_a:
        try:
            data_bytes = _build_session_bytes()
        except Exception:
            data_bytes = None
        st.download_button(
            "💾 Save",
            data=data_bytes if data_bytes is not None else b"{}",
            file_name="afmo_session.json",
            mime="application/json",
            key=f"{key_prefix}__session_download",
            help="Download your current session as a JSON file (data, settings, forecasts, evaluation, selections, UI theme).",
            use_container_width=True,
            disabled=(data_bytes is None),
        )
        if data_bytes is None:
            st.caption("*(Nothing to save yet — please import data.)*")

    with col_b:
        up = st.file_uploader("Load", type=["json"], label_visibility="collapsed", key=f"{key_prefix}__session_upload")
        if up is not None:
            msg = _apply_loaded_session(up)
            st.success(msg)
        
    with st.expander("📁 AFMo/examples"):
        try:
            sess_files = sorted([p for p in _EXAMPLES_DIR.glob("*.json")])
        except Exception:
            sess_files = []
        pick = st.selectbox("Load session JSON from examples", options=["-- select --"] + [p.name for p in sess_files], key="ex_sel_sess")
        if st.button("Load selected session"):
            if pick and pick != "-- select --":
                p = _EXAMPLES_DIR / pick
                try:
                    with p.open("rb") as fh:
                        msg = _apply_loaded_session(fh)
                    st.success(f"Loaded: {p.name}")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
        if st.button("💾 Save current session to examples"):
            try:
                b = _build_session_bytes()
                outp = _EXAMPLES_DIR / "afmo_session.json"
                with outp.open("wb") as fh:
                    fh.write(b or b"{}")
                st.success(f"Saved to {outp}")
            except Exception as e:
                st.error(f"Failed to save: {e}")