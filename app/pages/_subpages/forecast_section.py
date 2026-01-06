"""Subpage: Forecast section"""

def run_forecast_section():
    import _bootstrap
    import streamlit as st
    import pandas as pd
    import numpy as np
    from utils.theme_utils import theme_header
    from utils.io_utils import _get_selected_series_list, infer_m_from_freq
    from afmo.fc_models import get_fc_result

    MODEL_DESCRIPTIONS = {
        "AUTOARIMA": (
            "<strong>AUTOARIMA</strong><br>"
            "Automatically selects an ARIMA or SARIMA model based on an information criterion "
            "(e.g., AIC, BIC, HQIC).<br><br>"
            "<strong>Key parameters:</strong><br>"
            "• <code>seasonal</code>: Enables seasonal modeling<br>"
            "• <code>m</code>: Seasonal period length<br>"
            "• <code>max_p, max_d, max_q</code>: Upper bounds for non-seasonal orders<br>"
            "• <code>max_P, max_D, max_Q</code>: Upper bounds for seasonal orders<br>"
            "• <code>information criterion</code>: Metric used for model selection"
        ),

        "ARIMA": (
            "<strong>ARIMA</strong><br>"
            "Autoregressive Integrated Moving Average model for non-seasonal time series "
            "with manually specified parameters.<br><br>"
            "<strong>Key parameters:</strong><br>"
            "• <code>p</code>: Autoregressive order<br>"
            "• <code>d</code>: Degree of differencing<br>"
            "• <code>q</code>: Moving average order"
        ),

        "SARIMA": (
            "<strong>SARIMA</strong><br>"
            "Seasonal extension of ARIMA that models repeating seasonal patterns "
            "in time series data.<br><br>"
            "<strong>Key parameters:</strong><br>"
            "• <code>p, d, q</code>: Non-seasonal ARIMA orders<br>"
            "• <code>P, D, Q</code>: Seasonal ARIMA orders<br>"
            "• <code>m</code>: Length of the seasonal cycle"
        ),

        "ETS": (
            "<strong>ETS</strong><br>"
            "Exponential Smoothing model based on Error, Trend, and Seasonality components. "
            "Automatically selects a suitable configuration.<br><br>"
            "<strong>Key parameters:</strong><br>"
            "• <code>trend</code>: Type of trend component (additive/multiplicative)<br>"
            "• <code>seasonal</code>: Type of seasonal component<br>"
            "• <code>m</code>: Seasonal period length"
        ),
    }

    st.markdown("""
    <style>
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
    .help-tooltip { position: relative; display: inline-block; }
    .help-tooltip .tooltip-text {
    visibility: hidden;
    width: 340px;
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
    .help-tooltip:hover .tooltip-text { visibility: visible; }
    </style>
    """, unsafe_allow_html=True)

    # Page header with theme & session controls
    theme_header("Forecast & Accuracy", key="hdr_forecast")

    from utils.state_utils import ensure_state
    from afmo.plot import forecast_plot as _plot_fc

    ensure_state()

    st.session_state.setdefault("forecast_sigs_current", set())
    st.session_state.setdefault("forecast_sigs_prev", set())

    def make_sig(mode: str, series: str, horizon: int, params: dict) -> str:
        # stable representation (no heavy objects)
        items = tuple(sorted(params.items()))
        return f"{mode}|{series}|h={int(horizon)}|{items}"
    
    def sig_exists(sig: str) -> bool:
        return sig in (st.session_state["forecast_sigs_prev"] | st.session_state["forecast_sigs_current"])



    def start_new_forecast_run():
        """Move the current batch into previous, and clear current batch."""
        st.session_state.setdefault("forecasts", {})
        st.session_state.setdefault("forecasts_current", {})
        st.session_state.setdefault("forecast_sigs_prev", set())
        st.session_state.setdefault("forecast_sigs_current", set())

        cur = st.session_state.get("forecasts_current", {})
        if cur:
            # Move everything from current → previous
            for old_key, old_fc in cur.items():
                if isinstance(old_fc, pd.DataFrame) and not old_fc.empty:
                    k = old_key
                    n = 2
                    while k in st.session_state["forecasts"]:
                        k = f"{old_key} ({n})"
                        n += 1
                    st.session_state["forecasts"][k] = old_fc.copy()

        st.session_state["forecast_sigs_prev"] |= st.session_state["forecast_sigs_current"]
        st.session_state["forecast_sigs_current"] = set()

        st.session_state["forecasts_current"] = {}


    def store_current_forecast(key: str, fc: pd.DataFrame):
        """Store a forecast into the current batch."""
        st.session_state.setdefault("forecasts_current", {})
        st.session_state["forecasts_current"][key] = fc.copy()


    
    # transform string frequencies such as W-MON to integer frequencies
    freq = st.session_state.get("freq")
    _data = st.session_state.get("data")
    y_idx = _data.index if isinstance(_data, pd.DataFrame) else None
    m_default = infer_m_from_freq(freq, y_index=y_idx, default=1)

    st.title("Forecast")

    # get uploaded data set
    data = st.session_state['data']

    if data is None:
        st.warning("Please load data on the Import page first.")
        return

    # Series selection
    default_fallback = st.session_state.get("active_col")
    if default_fallback not in data.columns:
        default_fallback = list(data.columns)[0]
    # pull from Data page (if any), else fallback
    sel_series = _get_selected_series_list(data, default_fallback)
    sel_series = st.multiselect(
        "Target series",
        options=list(data.columns),
        default=sel_series,
        key="forecast_series_multiselect",
    )
    if not sel_series:
        st.warning("Select at least one series.")
        return
    # keep global state in sync so Evaluation/Data stay consistent
    st.session_state["selected_cols"] = sel_series
    sel_series = _get_selected_series_list(data, default_fallback)

    st.markdown("---")

    # Global controls
    left, right = st.columns([1, 1])
    with left:
        from afmo.core.registry import FC_MODELS
        available_modes = [n for n in FC_MODELS.keys()][:-1]  # exclude LightGBM for now
        if not available_modes:
            available_modes = ["AUTOARIMA"]

        # Label + tooltip right next to it (explanation depends on current selection)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:6px; margin-bottom: 0.25rem;">
            <strong>Forecast Method</strong>
            <span class="help-tooltip">
                <span class="help-icon">?</span>
                <span class="tooltip-text">
                {MODEL_DESCRIPTIONS.get(st.session_state.get("forecast_mode", available_modes[0]),
                                        "Select a method to see details.")}
                </span>
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode = st.selectbox(
            label="Forecast Method",  
            options=available_modes,
            index=0,
            key="forecast_mode",
            label_visibility="collapsed",
        )
    with right:
        c_h = st.number_input("Forecast horizon (steps)", min_value=2, max_value=10000, value=int(st.session_state.get("fc_horizon", 12)), step=1, key="forecast_horizon")

    # AUTO-ARIMA
    if mode == "AUTOARIMA":
        with st.expander("AUTO-ARIMA settings", expanded=True):
            seasonal = st.checkbox("Allow seasonality", value=False, key="auto_seasonal")
            m = st.number_input("Seasonal period m (if seasonal)", 1, 365, int(m_default), step=1, key="auto_m")
            max_p = st.number_input("max p", 0, 10, 5, step=1, key="auto_max_p")
            max_d = st.number_input("max d", 0, 5, 2, step=1, key="auto_max_d")
            max_q = st.number_input("max q", 0, 10, 5, step=1, key="auto_max_q")
            max_P = st.number_input("max P", 0, 10, 2, step=1, key="auto_max_P")
            max_D = st.number_input("max D", 0, 5, 1, step=1, key="auto_max_D")
            max_Q = st.number_input("max Q", 0, 10, 2, step=1, key="auto_max_Q")
            ic = st.selectbox("Information criterion", ["aic", "bic", "hqic", "oob"], index=0, key="auto_ic")
            with st.expander("What do these information criteria mean?", expanded=False):
                tab_aic, tab_bic, tab_hqic, tab_oob = st.tabs(["AIC", "BIC", "HQIC", "OOB"])

                with tab_aic:
                    st.markdown(
                        "**AIC (Akaike Information Criterion)**  \n"
                        "- Balances model fit and complexity  \n"
                        "- Typically favors models with better predictive accuracy  \n"
                        "- Often selects slightly more complex models than BIC"
                    )

                with tab_bic:
                    st.markdown(
                        "**BIC (Bayesian Information Criterion)**  \n"
                        "- Penalizes model complexity more strongly than AIC  \n"
                        "- Often selects simpler models  \n"
                        "- Useful when you want to avoid overfitting"
                    )

                with tab_hqic:
                    st.markdown(
                        "**HQIC (Hannan–Quinn Information Criterion)**  \n"
                        "- A compromise between AIC and BIC  \n"
                        "- Penalizes complexity more than AIC, less than BIC"
                    )

                with tab_oob:
                    st.markdown(
                        "**OOB (Out-of-Bag / holdout-style estimate)**  \n"
                        "- Estimates predictive performance on unseen data (if supported)  \n"
                        "- Can behave differently depending on the underlying implementation"
                    )

        if st.button("Fit AUTO-ARIMA & Forecast", key="btn_auto_arima", disabled=False):
            params = {
                "seasonal": bool(seasonal),
                "m": int(m) if seasonal and m > 0 else 1,
                "max_p": int(max_p), "max_d": int(max_d), "max_q": int(max_q),
                "max_P": int(max_P), "max_D": int(max_D), "max_Q": int(max_Q),
                "ic": str(ic),
            }

            to_fit = []
            for series in sel_series:
                sig = make_sig("AUTOARIMA", series, int(c_h), params)
                if not sig_exists(sig):
                    to_fit.append((series, sig))

            if not to_fit:
                st.info("Skipping: all selected AUTOARIMA forecasts with these settings already exist.")
            else:
                start_new_forecast_run()
                for series, sig in to_fit:
                    with st.spinner(f"Fitting AUTO-ARIMA for {series}…"):
                        y_train = data[series].dropna()
                        dict_res = get_fc_result(
                            name="AUTOARIMA", y=y_train, steps=int(c_h), seasonal=seasonal,
                            m=int(m) if seasonal and m > 0 else 1,
                            max_p=int(max_p), max_d=int(max_d), max_q=int(max_q),
                            max_P=int(max_P), max_D=int(max_D), max_Q=int(max_Q),
                            information_criterion=ic
                        )

                        key = dict_res["model_key"]
                        fc = dict_res["forecast"]
                        res = dict_res["model"]

                        st.session_state["model_name"][key] = mode
                        st.session_state["model_params"][key] = params
                        st.session_state["models"][key] = res

                        store_current_forecast(key, fc)
                        st.session_state["forecast_sigs_current"].add(sig)

                        st.success(f"Fitted current forecast: {key}")

                    if st.session_state.get("_active_page") == "Forecast & Accuracy":
                        st.plotly_chart(_plot_fc(data, series, fc, title=key),
                                        use_container_width=True,
                                        key=f"plot_fit::{key}")

    # SARIMA
    if mode == "SARIMA":
        with st.expander("SARIMA settings", expanded=True):
            p = st.number_input("p", 0, 10, 1, step=1, key="sarima_p")
            d = st.number_input("d", 0, 5, 1, step=1, key="sarima_d")
            q = st.number_input("q", 0, 10, 1, step=1, key="sarima_q")
            P = st.number_input("P", 0, 10, 0, step=1, key="sarima_P")
            D = st.number_input("D", 0, 5, 0, step=1, key="sarima_D")
            Q = st.number_input("Q", 0, 10, 0, step=1, key="sarima_Q")
            m = st.number_input("m (seasonal period)", 0, 365, int(m_default), step=1, key="sarima_m")

        if st.button("Fit SARIMA & Forecast", key="btn_sarima"):
            params = {
                "order": (int(p), int(d), int(q)),
                "seasonal_order": (int(P), int(D), int(Q), int(m)),
            }

            to_fit = []
            for series in sel_series:
                sig = make_sig("SARIMA", series, int(c_h), params)
                if not sig_exists(sig):
                    to_fit.append((series, sig))

            if not to_fit:
                st.info("Skipping: all selected SARIMA forecasts with these settings already exist.")
            else:
                start_new_forecast_run()
                for series, sig in to_fit:
                    with st.spinner(f"Fitting SARIMA for {series}…"):
                        y_train = data[series].dropna()
                        dict_res = get_fc_result(
                            name="SARIMA", y=y_train, steps=int(c_h),
                            order=params["order"],
                            seasonal_order=params["seasonal_order"]
                        )

                        key = dict_res["model_key"]
                        fc = dict_res["forecast"]
                        res = dict_res["model"]

                        st.session_state["models"][key] = res
                        st.session_state["model_name"][key] = mode
                        st.session_state["model_params"][key] = params

                        store_current_forecast(key, fc)
                        st.session_state["forecast_sigs_current"].add(sig)

                        st.success(f"Fitted current forecast: {key}")

                    if st.session_state.get("_active_page") == "Forecast & Accuracy":
                        st.plotly_chart(_plot_fc(data, series, fc, title=key),
                                        use_container_width=True,
                                        key=f"plot_fit::{key}")



    # ARIMA
    if mode == "ARIMA":
        with st.expander("ARIMA settings", expanded=True):
            p = st.number_input("p", 0, 10, 1, step=1, key="arima_p")
            d = st.number_input("d", 0, 5, 1, step=1, key="arima_d")
            q = st.number_input("q", 0, 10, 1, step=1, key="arima_q")

        if st.button("Fit ARIMA & Forecast", key="btn_arima"):
            params = {"order": (int(p), int(d), int(q))}

            to_fit = []
            for series in sel_series:
                sig = make_sig("ARIMA", series, int(c_h), params)
                if not sig_exists(sig):
                    to_fit.append((series, sig))

            if not to_fit:
                st.info("Skipping: all selected ARIMA forecasts with these settings already exist.")
            else:
                start_new_forecast_run()
                for series, sig in to_fit:
                    with st.spinner(f"Fitting ARIMA for {series}…"):
                        y_train = data[series].dropna()
                        dict_res = get_fc_result(
                            name="ARIMA", y=y_train, steps=int(c_h),
                            order=params["order"]
                        )

                        key = dict_res["model_key"]
                        fc = dict_res["forecast"]
                        res = dict_res["model"]

                        st.session_state["models"][key] = res
                        st.session_state["model_name"][key] = mode
                        st.session_state["model_params"][key] = params

                        store_current_forecast(key, fc)
                        st.session_state["forecast_sigs_current"].add(sig)

                        st.success(f"Fitted current forecast: {key}")

                    if st.session_state.get("_active_page") == "Forecast & Accuracy":
                        st.plotly_chart(_plot_fc(data, series, fc, title=key),
                                        use_container_width=True,
                                        key=f"plot_fit::{key}")


    # ETS
    if mode == "ETS":
        with st.expander("ETS settings", expanded=True):
            st.caption("AUTO will pick a suitable ETS configuration.")
            trend = st.selectbox("Trend", [None, "add", "mul"], index=0, key="ets_trend")
            seasonal = st.selectbox("Seasonal", [None, "add", "mul"], index=0, key="ets_seasonal")
            m = st.number_input("Seasonal period m", 0, 365, int(m_default), step=1, key="ets_m")

        if st.button("Fit ETS (AUTO) & Forecast", key="btn_ets_auto"):
            params = {
                "trend": trend,
                "seasonal": seasonal,
                "m": int(m),
            }

            to_fit = []
            for series in sel_series:
                sig = make_sig("ETS", series, int(c_h), params)
                if not sig_exists(sig):
                    to_fit.append((series, sig))

            if not to_fit:
                st.info("Skipping: all selected ETS forecasts with these settings already exist.")
            else:
                start_new_forecast_run()
                for series, sig in to_fit:
                    with st.spinner(f"Fitting ETS (AUTO) for {series}…"):
                        y_train = data[series].dropna()
                        dict_res = get_fc_result(
                            name="ETS", y=y_train, steps=int(c_h),
                            trend=trend, seasonal=seasonal, m=int(m)
                        )

                        key = dict_res["model_key"]
                        fc = dict_res["forecast"]
                        res = dict_res["model"]

                        st.session_state["models"][key] = res
                        st.session_state["model_name"][key] = mode
                        st.session_state["model_params"][key] = params

                        store_current_forecast(key, fc)
                        st.session_state["forecast_sigs_current"].add(sig)

                        st.success(f"Fitted current forecast: {key}")

                    if st.session_state.get("_active_page") == "Forecast & Accuracy":
                        st.plotly_chart(_plot_fc(data, series, fc, title=key),
                                        use_container_width=True,
                                        key=f"plot_fit::{key}")


    st.markdown("---")

    # Quick list of previous keys
    # NEEDED TO BE CHANGED SINCE WE CURRENTLY HAVE TWO SPOTS WHERE FORECASTS ARE BEING SAVED
    _prev_keys = list(st.session_state.get("forecasts", {}).keys() | st.session_state.get("forecasts_current",{}).keys())
    if _prev_keys:
        with st.expander("Saved forecast keys", expanded=False):
            st.write(", ".join(sorted(_prev_keys)))

    st.subheader("Current fitted forecasts")

    _cur_fcs = st.session_state.get("forecasts_current", {})
    if _cur_fcs:
        for _key, _fc in _cur_fcs.items():
            try:
                _series = _key.split("|")[-1].strip() if "|" in _key else st.session_state.get("active_col")
                if _series and _series in data.columns and isinstance(_fc, pd.DataFrame) and not _fc.empty:
                    st.plotly_chart(_plot_fc(data, _series, _fc, title=_key), use_container_width=True)
            except Exception as _e:
                st.warning(f"Could not render current forecast '{_key}': {_e}")
    else:
        st.caption("No current forecasts yet. Fit a model above to create them.")

    _prev_fcs = st.session_state.get("forecasts", {})
    _model_name = st.session_state.get("model_name", {})

    if _prev_fcs:
        prev_keys = list(_prev_fcs.keys())

        def _series_from_key(k: str) -> str:
            return k.split("|")[-1].strip() if "|" in k else ""

        prev_series_options = sorted({s for s in (_series_from_key(k) for k in prev_keys) if s})
        prev_model_options = sorted({(_model_name.get(k) or "Unknown") for k in prev_keys})

        with st.expander("Previously fitted forecasts", expanded=False):
            # Filters
            fcol1, fcol2 = st.columns([2, 1])

            with fcol1:
                selected_prev_series = st.multiselect(
                    "Select one or more series",
                    options=prev_series_options,
                    key="prev_fc_series_filter",
                )
            with fcol2:
                selected_prev_model = st.selectbox(
                    "Forecast model",
                    options=["All"] + prev_model_options,
                    index=0,
                    key="prev_fc_model_filter",
                )

            # Apply filters
            filtered_items = []
            for k, fc in _prev_fcs.items():
                s = _series_from_key(k)
                m = _model_name.get(k) or "Unknown"

                if selected_prev_series and s not in selected_prev_series:
                    continue
                if selected_prev_model != "All" and m != selected_prev_model:
                    continue

                filtered_items.append((k, fc, s, m))

            if not filtered_items:
                st.info("No forecasts match the selected filters.")
            else:
                st.caption(f"Showing {len(filtered_items)} previous forecast(s).")

                for k, fc, s, m in filtered_items:
                    try:
                        if s and s in data.columns and isinstance(fc, pd.DataFrame) and not fc.empty:
                            st.plotly_chart(_plot_fc(data, s, fc, title=f"{k}"), use_container_width=True)
                    except Exception as _e:
                        st.warning(f"Could not render stored forecast '{k}': {_e}")
    else:
        st.caption("No previous forecasts yet. Fit at least two runs to see history here.")


