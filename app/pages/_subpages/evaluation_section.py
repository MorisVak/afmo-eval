
"""Subpage: Evaluation section."""
import os
import html

def run_evaluation_section():
    import streamlit as st
    import pandas as pd
    from utils.state_utils import ensure_state
    from afmo.predictors import get_pred_value
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
    width: 360px;
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
                
    /* Allow tooltips to escape table layout */
    .eval-table-wrap { width: 100%; overflow: visible !important; }
    .eval-table { border-collapse: collapse; width: 100%; table-layout: fixed; overflow: visible !important; }
    .eval-th, .eval-td { padding: 6px 8px; overflow: visible !important; position: relative; }

    /* Make sure tooltip sits on top of everything */
    .help-tooltip { position: relative; display: inline-block; overflow: visible !important; }
    .help-tooltip .tooltip-text {
    z-index: 999999;
    }
    </style>
    """, unsafe_allow_html=True)
    SCORE_TOOLTIPS = {
    "Predictability": (
        "<strong>Predictability</strong><br>"
        "Measures how structured and predictable the forecast residuals are.<br>"
        "Lower randomness and higher temporal structure indicate higher predictability."
    ),
    "Effectiveness": (
        "<strong>Effectiveness</strong><br>"
        "Measures how well the forecasting model removes systematic patterns from the data.<br>"
        "Residuals closer to white noise indicate higher effectiveness."
    ),
}

    METRIC_TOOLTIPS = {
        # Predictability metrics
        "rpa": (
            "<strong>RPA – Residual Predictability Analysis</strong><br>"
            "Quantifies the degree of predictability remaining in the residuals."
        ),
        "rqa": (
            "<strong>RQA – Recurrence Quantification Analysis</strong><br>"
            "Measures recurring patterns and temporal structure in the residual series."
        ),
        "mia": (
            "<strong>MIA – Mutual Information Analysis</strong><br>"
            "Captures nonlinear dependencies in the residuals via information-theoretic measures."
        ),

        # Effectiveness metrics
        "bds": (
            "<strong>BDS Test</strong><br>"
            "Statistical test for independence and nonlinearity in the residuals."
        ),
        "ljb": (
            "<strong>Ljung–Box Test</strong><br>"
            "Tests whether residuals are free from autocorrelation (white-noise assumption)."
        ),
        "runs": (
            "<strong>Runs Test</strong><br>"
            "Tests randomness of residual sign changes over time."
        ),
}

    EVAL_DESCRIPTIONS = {
    "In-sample (fitted values)": (
        "<strong>In-sample</strong><br>"
        "Evaluates the model on the same data used for fitting (fitted values).<br>"
        "Useful for checking overall fit, but may be optimistic."
    ),
    "Backtest": (
        "<strong>Backtest</strong><br>"
        "Hold out the last <code>k</code> observations and evaluate forecasts against that holdout.<br>"
        "Gives a more realistic estimate of performance on unseen data."
    ),
    "Rolling CV": (
        "<strong>Rolling CV</strong><br>"
        "Rolling-origin evaluation: repeatedly fit on an expanding window and forecast the next horizon.<br>"
        "Captures performance stability over time."
    ),
    "Similarity CV": (
        "<strong>Similarity CV</strong><br>"
        "Feature-based cross-validation that leverages similarity between series/windows in the dataset.<br>"
        "Designed for multi-series settings."
    ),
    "Cl_KMeans_CV": (
        "<strong>KMeans Clustering CV</strong><br>"
        "Groups series/windows using KMeans on extracted features and evaluates within/against clusters.<br>"
        "Useful when series form distinct patterns."
    ),
    "Cl_Hier_CV": (
        "<strong>Hierarchical Clustering CV</strong><br>"
        "Clusters series/windows hierarchically based on features and evaluates cluster-aware splits."
    ),
    "Cl_Density_CV": (
        "<strong>Density Clustering CV</strong><br>"
        "Uses density-based clustering on features to identify groups/outliers for evaluation splits."
    ),
    "Meta-Learning": (
        "<strong>Meta-Learning</strong><br>"
        "Uses a trained meta-model to estimate evaluation metrics based on dataset/model characteristics.<br>"
        "Can speed up evaluation when full CV is expensive."
    ),
    "AUTO": (
        "<strong>AUTO</strong><br>"
        "Automatically selects the best-performing evaluation method based on cross-validated data."
    ),
}
    def _format_percentages(x, decimals: int = 2) -> str:
        """Format numeric values as percentages.
        Assumes values are proportions in [0, 1] (common for these scores).
        """
        try:
            v = float(x)
        except Exception:
            return "—"
        if pd.isna(v):
            return "—"
        return f"{v * 100:.{decimals}f}%"

    
    def header_with_tooltip(title: str, tooltip_html: str):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:6px; margin: 0.75rem 0 0.5rem;">
            <h3 style="margin:0;">{title}</h3>
            <span class="help-tooltip">
                <span class="help-icon">?</span>
                <span class="tooltip-text">{tooltip_html}</span>
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    ensure_state()

    def _parse_model_key(key: str):
        """Return (horizon:int, series_name:str) from a model key like 'ARIMA(...) | 12 | series'."""
        parts = [p.strip() for p in key.split("|")]
        if len(parts) >= 2:
            try:
                h = int(parts[-2])
            except Exception:
                h = int(st.session_state.get("fc_horizon", 12))
            s = parts[-1]
            return h, s
        # Fallbacks
        return int(st.session_state.get("fc_horizon", 12)), (st.session_state.get("active_col") or "")

    def _render_eval_results(sel_models, label):

        if "predictions" not in st.session_state:
            return

        # Group defenitions
        GROUPS = [
            ("Predictability", "predictability", ["rpa", "rqa", "mia"]),
            ("Effectiveness", "effectiveness", ["bds", "ljb", "runs"]),
        ]

        # helper methods
        def _to_df(dict_res) -> pd.DataFrame:
            """Normalize dict_res into a DataFrame with index=metric and cols=[mean, lower, upper]."""
            try:
                df = pd.DataFrame({k: v.iloc[0] for k, v in dict_res.items()}).T
            except Exception:
                df = pd.DataFrame(dict_res)

            df.index.name = "metric"

            wanted = ["mean", "lower", "upper"]
            cols = [c for c in wanted if c in df.columns]
            if cols:
                df = df[cols]
            return df

        def _render_score_row(title: str, row: pd.Series):
            c1, c2, c3 = st.columns(3)
            for col, key in zip((c1, c2, c3), ("mean", "lower", "upper")):
                if key in row.index:
                    col.metric(key, _format_percentages(row[key], decimals=2))
                else:
                    col.metric(key, "—")

        def _render_metrics_table(df: pd.DataFrame, metric_keys: list[str]):
            present = [m for m in metric_keys if m in df.index]
            if not present:
                st.info("No metrics available for this block.")
                return

            view = df.loc[present]

            rows_html = ""
            for metric in present:
                tip = METRIC_TOOLTIPS.get(metric, "")
                label = html.escape(metric)

                if tip:
                    metric_cell = f"""
                    <span style="display:inline-flex; align-items:center; gap:6px;">
                        <strong>{label}</strong>
                        <span class="help-tooltip">
                            <span class="help-icon">?</span>
                            <span class="tooltip-text">{tip}</span>
                        </span>
                    </span>
                    """
                else:
                    metric_cell = f"<strong>{label}</strong>"

                vals = "".join(
                    f"<td class='eval-td'>{_format_percentages(view.loc[metric, c], decimals=2)}</td>"
                    for c in view.columns
                )

                rows_html += f"""
                <tr>
                <td class='eval-td'>{metric_cell}</td>
                {vals}
                </tr>
                """

            header_cols = "".join(
                f"<th class='eval-th' style='text-align:left;'>{html.escape(c)}</th>"
                for c in view.columns
            )

            st.markdown(
                f"""
                <div class="eval-table-wrap">
                <table class="eval-table">
                    <thead>
                    <tr>
                        <th class="eval-th" style="text-align:left;">Metric</th>
                        {header_cols}
                    </tr>
                    </thead>
                    {rows_html}
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Model filtering UI (search + expand all)
        predictions = st.session_state.get("predictions", {})
        available_models = sorted(predictions.keys())

        ms_key = f"eval_model_multiselect_{label}"

        if ms_key not in st.session_state:
            st.session_state[ms_key] = [] 
        else:
            # Keep selection valid if models list changed
            st.session_state[ms_key] = [m for m in st.session_state[ms_key] if m in available_models]

        selected_models = st.multiselect(
            "Select models (leave empty to show all)",
            options=available_models,
            key=ms_key,           
        )

        expand_all_models = st.toggle(
            "Expand all models",
            value=False,
            key=f"eval_expand_all_{label}",
        )

        #If no mode is selected we just show every model
        models_to_render = selected_models if len(selected_models) > 0 else available_models

        #Render per model in expanders 
        for model in models_to_render:
            pred_map = st.session_state["predictions"].get(model, {})
            if label not in pred_map:
                continue

            dict_res = pred_map[label]
            df = _to_df(dict_res)

            with st.expander(str(model), expanded=expand_all_models):

                # Predictability + Effectiveness blocks
                for block_title, score_key, metric_keys in GROUPS:
                    with st.container(border=True):
                        # Score row (mean/lower/upper)
                        # Block title with tooltip
                        tooltip = SCORE_TOOLTIPS.get(block_title)
                        if tooltip:
                            header_with_tooltip(block_title, tooltip)
                        else:
                            st.markdown(f"{block_title}")

                        # Score row
                        if score_key in df.index:
                            _render_score_row("", df.loc[score_key])
                        else:
                            st.warning(f"Missing '{score_key}' in results.")

                        # Details table
                        st.markdown("##### Details")
                        _render_metrics_table(df, metric_keys)

    # Data & available models
    data = st.session_state.get("data", None)
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        st.title("Evaluation")
        st.info("Load data and run at least one forecast first.")
        return

    models = st.session_state.get("models", {})
    if not models:
        st.title("Evaluation")
        st.info("No fitted models found. Go to Forecast and fit a model.")
        return

    st.title("Evaluation")

    # Model multi-select
    model_keys = list(models.keys())
    sel_models = st.multiselect(
        "Target models",
        options=model_keys,
        default=model_keys,
        key="forecast_models_multiselect"
    )
    if not sel_models:
        st.warning("Select at least one model.")
        return

    # Persist active tab via radio
    tab_options = ["In-sample (fitted values)", "Backtest", "Rolling CV", "Similarity CV",
                "Cl_KMeans_CV", "Cl_Hier_CV", "Cl_Density_CV", "Meta-Learning", "AUTO"]

    # Custom label + tooltip (shows explanation for currently selected method)
    current_choice = st.session_state.get("active_eval_tab", tab_options[0])

    active_tab = st.radio(
        "View",
        tab_options,
        index=tab_options.index(current_choice),
        horizontal=True,
        key="eval_view_switch",
        label_visibility="collapsed",
    )

    st.session_state.active_eval_tab = active_tab

    # Convenience handles
    model_name_map = st.session_state.get("model_name", {})
    model_params_map = st.session_state.get("model_params", {})
    #due to the current and previous forecast logic we need to fetch both states.
    forecasts_map = {}
    forecasts_map.update(st.session_state.get("forecasts", {}))
    forecasts_map.update(st.session_state.get("forecasts_current", {}))
    st.session_state.setdefault("predictions", {})
    # In-sample
    if active_tab == "In-sample (fitted values)":
        header_with_tooltip(
        "In-sample evaluation",
        EVAL_DESCRIPTIONS["In-sample (fitted values)"]
        )
        if st.button("Perform In-sample fit", key="btn_insample", disabled=False):
            with st.spinner("Computing In-sample fit…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    y_pred = forecasts_map.get(model, {}).get("is")
                    if y_pred is None:
                        continue

                    dict_res, _ = get_pred_value(
                        name="insample",
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )

                    st.session_state.setdefault("predictions", {}).setdefault(model, {})["In-sample fit"] = dict_res
                    computed += 1

            st.success(f"In-sample fit computed for {computed} model(s).")
        _render_eval_results(sel_models, "In-sample fit")

    # Backtest
    if active_tab == "Backtest":
        header_with_tooltip(
        "Backtest with holdout k",
        EVAL_DESCRIPTIONS["Backtest"]
        )
        if st.button("Perform Backtest", key="btn_backtest", disabled=False):
            with st.spinner("Computing Backtest…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name].iloc[:-int(h)]
                    dict_res, _ = get_pred_value(
                        name='backtest_k',
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"Backtest": dict_res})
                    computed += 1
            st.success(f"Backtest computed for {computed} model(s).")
        _render_eval_results(sel_models, "Backtest")

    # Rolling CV
    if active_tab == "Rolling CV":
        header_with_tooltip(
        "Rolling origin CV",
        EVAL_DESCRIPTIONS["Rolling CV"]
        )
        if st.button("Perform RollingCV", key="btn_rolling_cv", disabled=False):
            with st.spinner("Computing RollingCV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    dict_res, _ = get_pred_value(
                        name='rolling_cv',
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"RollingCV": dict_res})
                    computed +=1 
            st.success(f"RollingCV computed for {computed} model(s).")
        _render_eval_results(sel_models, "RollingCV")

    # Simulated CV
    if active_tab == "Similarity CV":
        header_with_tooltip(
        "Feature-based Similarity CV based on cross-validated data set",
        EVAL_DESCRIPTIONS["Similarity CV"]
        )
        if st.button("Perform Similarity CV", key="btn_sim_cv", disabled=False):
            with st.spinner("Computing Similarity CV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    dict_res, _ = get_pred_value(
                        name='sim_cv',
                        Y=data,
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"SimCV": dict_res})
                    computed += 1
            st.success(f"Similarity CV computed for {computed} model(s).")
        _render_eval_results(sel_models, "SimCV")


    # Clustering KMeans CV
    if active_tab == "Cl_KMeans_CV":
        header_with_tooltip(
        "Feature-based KMeans Clustering based on cross-validated data set",
        EVAL_DESCRIPTIONS["Cl_KMeans_CV"]
        )
        if st.button("Perform Clustering KMeans CV", key="btn_cl_kmeans_cv", disabled=False):
            with st.spinner("Computing Clustering KMeans CV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    dict_res, _ = get_pred_value(
                        name='cl_kmeans_cv',
                        Y=data,
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"ClKMeansCV": dict_res})
                    computed += 1
            st.success(f"Clustering KMeans CV computed for {computed} model(s).")
        _render_eval_results(sel_models, "ClKMeansCV")

        # Clustering KMeans CV
    if active_tab == "Cl_Hier_CV":
        header_with_tooltip(
        "Feature-based Hierarchical Clustering based on cross-validated data set",
        EVAL_DESCRIPTIONS["Cl_Hier_CV"]
        )
        if st.button("Perform Clustering Hierarchical CV", key="btn_cl_hier_cv", disabled=False):
            with st.spinner("Computing Clustering Hierarchical CV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    dict_res, _ = get_pred_value(
                        name='cl_hier_cv',
                        Y=data,
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"ClHierCV": dict_res})
                    computed += 1
            st.success(f"Clustering Hierarchical CV computed for {computed} model(s).")
        _render_eval_results(sel_models, "ClHierCV")

        # Clustering KMeans CV
    if active_tab == "Cl_Density_CV":
        header_with_tooltip(
        "Feature-based Density Clustering based on cross-validated data set",
        EVAL_DESCRIPTIONS["Cl_Density_CV"]
        )
        if st.button("Perform Clustering Density CV", key="btn_cl_density_cv", disabled=False):
            with st.spinner("Computing Clustering Density CV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]
                    dict_res, _ = get_pred_value(
                        name='cl_density_cv',
                        Y=data,
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"ClDensityCV": dict_res})
                    computed += 1
            st.success(f"Clustering Density CV computed for {computed} model(s).")
        _render_eval_results(sel_models, "ClDensityCV")

        #Meta-Learning
    if active_tab == "Meta-Learning":
        header_with_tooltip(
        "Meta-Learning",
        EVAL_DESCRIPTIONS["Meta-Learning"]
        )

        pre_trained_mode_string = "Load pretrained"
        train_new_mode_string = "Train new"
        
        # Check if we just finished training and should switch to load mode
        if st.session_state.get("_switch_to_load_pretrained", False):
            st.session_state["ml_mode"] = pre_trained_mode_string
            st.session_state["_switch_to_load_pretrained"] = False
        
        # Initialize ml_mode if not set
        if "ml_mode" not in st.session_state:
            st.session_state["ml_mode"] = train_new_mode_string
        
        ml_mode = st.radio("Mode", [train_new_mode_string, pre_trained_mode_string], 
                          horizontal=True, key="ml_mode")

        if ml_mode == pre_trained_mode_string:
            # Check if we have a newly trained model to auto-load
            newly_trained_path = st.session_state.get("_newly_trained_model_path")
            if newly_trained_path and os.path.exists(newly_trained_path):
                # Auto-load the newly trained model
                st.success(f"Auto-loading newly trained model: {os.path.basename(newly_trained_path)}")
                # Store the path directly (preferred method)
                st.session_state["meta_model_file_path"] = newly_trained_path
                # Clear the flag
                st.session_state["_newly_trained_model_path"] = None
                # Auto-trigger loading
                st.session_state["_auto_load_meta_model"] = True
            
            # File uploader for manual uploads
            uploaded_file = st.file_uploader(
                "Upload Meta-Learning model file",
                type=["pkl", "joblib"],
                key="meta_model_file_uploader"
            )
            if uploaded_file is not None:
                # Store uploaded file data
                st.session_state["meta_model_file"] = uploaded_file
                st.session_state["meta_model_file_path"] = None
            
            # Show status and load button
            model_path = st.session_state.get("meta_model_file_path")
            model_file = st.session_state.get("meta_model_file")
            
            if model_path or model_file:
                if model_path:
                    st.success(f"Meta-Learning model loaded: {os.path.basename(model_path)}")
                    # Show download button for the loaded model
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="Download Meta-Model",
                            data=f,
                            file_name=os.path.basename(model_path),
                            mime="application/octet-stream",
                            key="download_meta_model_loaded"
                        )
                else:
                    st.success("Meta-Learning model file uploaded.")
                
                # Auto-load if flag is set, or show manual load button
                auto_load = st.session_state.get("_auto_load_meta_model", False)
                if auto_load or st.button("Load Meta-Learning Model", key="btn_load_meta_model", disabled=False):
                    st.session_state["_auto_load_meta_model"] = False
                    for model in sel_models:
                        h, series_name = _parse_model_key(model)
                        y = data[series_name]
                        
                        # Use path if available, otherwise use uploaded file
                        if model_path:
                            # Use path directly
                            meta_params = model_params_map.get(model, {})
                            if "__meta__" not in meta_params:
                                meta_params["__meta__"] = {}
                            meta_params["__meta__"]["meta_model_path"] = model_path
                            
                            dict_res, _ = get_pred_value(
                                name='meta_learning_regressor',
                                df=data,
                                y=y,
                                model_name=model_name_map.get(model, ""),
                                model_params=meta_params,
                                fc_horizon=int(h)
                            )
                        else:
                            # Use uploaded file
                            dict_res, _ = get_pred_value(
                                name='meta_learning_regressor',
                                df=data,
                                y=y,
                                model_name=model_name_map.get(model, ""),
                                model_params=model_params_map.get(model, {}),
                                fc_horizon=int(h),
                                meta_model_file=model_file
                            )
                        st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"MetaLearning": dict_res})
                    st.rerun()
                _render_eval_results(sel_models, "MetaLearning")
        else:
            st.subheader("Train a New Meta-Learning Model")
            st.info(
                "This process will train a new meta-model using the entire dataset. "
                "The trained model can then be used to generate evaluation metrics for all target series."
            )

            # Always show train button (cleared after training completes)
            if st.button("Train Meta-Model", key="btn_train_meta", disabled=False):
                    from afmo.meta_trainer import train_and_save_meta_model

                    # Create progress UI elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Use the first model as reference for model family
                    if sel_models:
                        model = sel_models[0]
                        h, series_name = _parse_model_key(model)

                        # Progress callback for UI updates
                        def update_progress(current, total, message):
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(f"{message} ({int(progress*100)}%)")

                        status_text.text("Starting meta-model training...")

                        # Train the meta-model using entire dataset
                        # Pass model_params from session (e.g., order/seasonal_order for SARIMA)
                        results = train_and_save_meta_model(
                            data=data,
                            target_model_family=model_name_map.get(model, "ARIMA"),
                            target_output_name=f"meta_model_{model}",
                            horizon=int(h),
                            ground_truth_mode="fast",
                            n_windows=3,
                            n_jobs=-1,  # Use all CPU cores
                            progress_callback=update_progress,
                            model_params=model_params_map.get(model, {})
                        )

                        # Store training results
                        st.session_state["meta_training_results"] = results
                        model_path = results.get('model_path')
                        st.session_state["meta_model_path"] = model_path

                        # Clear training state so user can train again
                        st.session_state["meta_model_trained"] = False
                        
                        # Set flags to switch to "Load pretrained" and auto-load the model
                        if model_path and os.path.exists(model_path):
                            st.session_state["_newly_trained_model_path"] = model_path
                            st.session_state["_switch_to_load_pretrained"] = True
                            st.session_state["_auto_load_meta_model"] = True

                        st.rerun()  # Refresh to switch tabs and load model

    # AUTO (BestMethod)
    if active_tab == "AUTO":
        header_with_tooltip(
        "Select best performing based evaluation method based on cross-validated data set.",
        EVAL_DESCRIPTIONS["AUTO"]
        )
        if st.button("Perform BestMethod", key="btn_best", disabled=False):
            with st.spinner("Computing Clustering Density CV…"):
                computed = 0
                for model in sel_models:
                    h, series_name = _parse_model_key(model)
                    y = data[series_name]#.iloc[:-int(h)]
                    dict_res, _ = get_pred_value(
                        name='best_method',
                        Y=data,
                        y=y,
                        model_name=model_name_map.get(model, ""),
                        model_params=model_params_map.get(model, {}),
                        fc_horizon=int(h)
                    )
                    st.session_state.setdefault("predictions", {}).setdefault(model, {}).update({"BestMethod": dict_res})
                    computed += 1
            st.success(f"Clustering Density CV computed for {computed} model(s).")
        _render_eval_results(sel_models, "BestMethod")
