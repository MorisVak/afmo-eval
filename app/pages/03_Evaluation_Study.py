import streamlit as st

from utils.state_utils import ensure_state
from utils.theme_utils import theme_header
from utils.persist_utils import pack_study_results, unpack_study_results
from afmo.plots.utils import apply_pretty_names_to_collector

from afmo.plots.build import build_all
from afmo.plots import PlotCollector
from afmo.plots.registry import PlotKind

from afmo.study import run_experiment, run_experiment_split
from afmo.helpers import freq_to_periods_per_year

st.session_state["_active_page"] = "Study"

try:
    import lightgbm as lgb
    st.write("LightGBM import OK:", lgb.__version__)
except Exception as e:
    st.error(f"LightGBM import FAILED: {type(e).__name__}: {e!s}")
    raise
# State
#"study_plot_kinds_selected"   
#"study_plot_kinds_built"      
#"plot_collector"              

def _ensure_study_plot_state() -> None:
    if "study_plot_kinds_selected" not in st.session_state:
        st.session_state["study_plot_kinds_selected"] = set()
    if "study_plot_kinds_built" not in st.session_state:
        st.session_state["study_plot_kinds_built"] = set()


def _header_export_button() -> None:
    """Render 'Download all plots' in the header (using the current collector)."""
    collector: PlotCollector | None = st.session_state.get("plot_collector")
    if not collector or not collector.all():
        st.button("Download all plots", disabled=True, help="No plots collected yet.")
        return
    try:
        zip_bytes = collector.as_zip_bytes(formats=("png","pdf"))
        st.download_button("Download all plots", data=zip_bytes, file_name="plots.zip", mime="application/zip")
    except Exception as e:
        st.button("Download all plots", disabled=True, help=f"Export error: {e}")


def _render_study_results(dict_res: dict, selected_kinds: list[PlotKind]) -> None:
    st.markdown("---")
    st.subheader("Plots")

    collector: PlotCollector | None = st.session_state.get("plot_collector")

    kind_label = {
        PlotKind.FEATURES_HIST: "Feature histograms",
        PlotKind.FC_EVAL_HIST: "Forecast-eval histograms",
        PlotKind.PRED_HIST: "Prediction histograms",
        PlotKind.VIOLIN_PRED: "Prediction violins (mean/lower/upper)",
        PlotKind.SCATTER_FEATURES: "Scatter of Features",
        PlotKind.SCATTER_CLUSTER: "Clustering assignment",
        PlotKind.HIST_CLUSTER: "Histogram of clusters",
        PlotKind.LINE_FEATURES: "Line plots of octant time series",
        PlotKind.PRED_EVAL_HIST: "Prediction-eval histograms",
        PlotKind.ASSOC_HEATMAPS: "Association heatmaps (ρ in [-1, 1])",
        PlotKind.BEST_METHOD_DIST: "Best method frequency",
        PlotKind.BEST_METHOD_QUADRANT: "Best-method evaluation",
        PlotKind.VIOLIN_PRED_EVAL: "Prediction-eval violins",
    }

    split_label = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }

    if not collector or not collector.all():
        st.info("No plots collected yet. Select plot types and click 'Plot selected'.")
        return

    # Build a quick index: kind_value -> split -> [artifacts...]
    arts_by_kind: dict[str, dict[str, list]] = {}
    for a in collector.all():
        arts_by_kind.setdefault(a.kind, {}).setdefault(a.split, []).append(a)

    # Render each requested plot kind in its own expander
    for kind in selected_kinds:
        kind_value = kind.value
        by_split = arts_by_kind.get(kind_value, None)
        if not by_split:
            continue

        title = kind_label.get(kind, kind_value)

        # Expander per plot kind
        with st.expander(title, expanded=False):
            # Always show three columns in a row (train/val/test)
            col_train, col_val, col_test = st.columns(3, gap="large")

            # Add visible split headers so the user knows what they see
            col_train.markdown(
                f"<h3 style='text-align: center; margin-top: 0;'>{split_label['train']}</h3>",
                unsafe_allow_html=True,
            )
            col_val.markdown(
                f"<h3 style='text-align: center; margin-top: 0;'>{split_label['val']}</h3>",
                unsafe_allow_html=True,
            )
            col_test.markdown(
                f"<h3 style='text-align: center; margin-top: 0;'>{split_label['test']}</h3>",
                unsafe_allow_html=True,
            )

            slot = {"train": col_train, "val": col_val, "test": col_test}

            # Render artifacts split-wise (keeps plots of same split in the correct column)
            for split in ("train", "val", "test"):
                arts = by_split.get(split, [])
                if not arts:
                    slot[split].caption("—")
                    continue

                for art in sorted(arts, key=lambda x: getattr(x, "title", "")):
                    c = slot[split]
                    if getattr(art, "title", None):
                        c.caption(art.title)

                    if getattr(art, "lib", "") == "plotly":
                        c.plotly_chart(art.fig, use_container_width=True)
                    else:
                        c.pyplot(art.fig, use_container_width=True)


def _show_study_results(dict_res: dict, selected_kinds: list[PlotKind]) -> None:
    pack = st.session_state.get("study_results_pack")
    if pack is None:
        st.session_state["study_results_pack"] = pack_study_results(dict_res)

    _render_study_results(dict_res, selected_kinds)

# Always restore raw Python types from whatever we have in session
def _load_study_results_from_session() -> dict | None:
    # If already a plain dict, just return it
    res = st.session_state.get("study_results", None)
    if isinstance(res, dict):
        return res

    # Otherwise try to unpack from the packed payload
    packed = st.session_state.get("study_results_pack", None)
    if packed is None:
        return None
    try:
        res = unpack_study_results(packed)
        # keep both forms in session for later reuse
        st.session_state["study_results"] = res
        st.session_state["study_results_pack"] = packed
        return res
    except Exception as e:
        st.warning(f"Failed to unpack study results: {e}")
        return None


def _page():
    try:
        from utils.persist_utils import pack_study_results, unpack_study_results
    except Exception:
        try:
            from afmo.utils.persist_utils import pack_study_results, unpack_study_results
        except Exception:
            pack_study_results = lambda x: x
            unpack_study_results = lambda x: x

    ensure_state()
    _ensure_study_plot_state()
    header_slot = st.empty()

    st.title("Study Evaluation")
    if "data" not in st.session_state or "freq" not in st.session_state:
        st.warning("Please load data & frequency first.")
    disabled_btn = not ("data" in st.session_state and "freq" in st.session_state)

    if st.button("Perform evaluation study", key="btn_study", disabled=disabled_btn):
        # set splitting True to calculate only parts of study.
        splitting = False
        model_name = "LightGBM" # "AUTOARIMA" #, "LightGBM"
        seasonal = True

        raw_freq = st.session_state.get("freq", None)
        if raw_freq in (None, "", "NONE", "None"):
            st.error("No frequency selected. Please select a valid time frequency before running the evaluation study.")
            return
        
        m = freq_to_periods_per_year(raw_freq)
        if splitting:
            # test how much splits we must do.
            if st.session_state.get("study_results", None) is None:
                # run for train
                with st.spinner("Running evaluation study..."):
                    dict_res = run_experiment_split(
                        df=st.session_state["data"],
                        split_name='train',
                        freq=m,
                        fc_horizon=26,
                        n=5,
                        model_name=model_name,
                        model_params={'m': m, 'seasonal': seasonal}
                    )
            elif 'val' not in list(st.session_state["study_results"]["splits"].keys()):
                # run for val
                with st.spinner("Running evaluation study..."):
                    dict_res = st.session_state["study_results"]
                    dict_res["splits"]["val"] = run_experiment_split(
                        df=st.session_state["data"],
                        split_name='val',
                        out=st.session_state["study_results"],
                        freq=m,
                        fc_horizon=26,
                        n=5,
                        model_name=model_name,
                        model_params={'m': m, 'seasonal': seasonal}
                    )["splits"]["val"]
            elif 'test' not in list(st.session_state["study_results"]["splits"].keys()):  
                # run for test
                with st.spinner("Running evaluation study..."):
                    dict_res = st.session_state["study_results"]
                    dict_res["splits"]["test"] = run_experiment_split(
                        df=st.session_state["data"],
                        split_name='test',
                        out=st.session_state["study_results"],
                        freq=m,
                        fc_horizon=26,
                        n=5,
                        model_name=model_name,
                        model_params={'m': m, 'seasonal': seasonal}
                    )["splits"]["test"]
            else:
                dict_res = st.session_state["study_results"]
        else:
            with st.spinner("Running evaluation study..."):
                # Progress display elements below spinner
                status_container = st.empty()
                progress_bar = st.progress(0.0)
                
                def update_progress(phase: str, message: str, progress: float):
                    """Callback to update UI progress."""
                    status_container.text(f"[{phase}] {message}")
                    progress_bar.progress(min(progress, 1.0))
                
                status_container.text("Initializing...")
                dict_res = run_experiment(
                    df=st.session_state["data"],
                    freq=m,
                    fc_horizon=26,
                    n=5,
                    model_name=model_name,
                    model_params={'m': m, 'seasonal': seasonal},
                    progress_callback=update_progress
                )
                
                # Clear progress elements after completion
                status_container.empty()
                progress_bar.empty()

        # Store both raw and packed forms in session
        st.session_state["study_results"] = dict_res
        try:
            st.session_state["study_results_pack"] = pack_study_results(dict_res)
        except Exception as e:
            st.warning(f"Packing study results failed, using raw only: {e}")
            st.session_state["study_results_pack"] = None
        # Show progress to the user
        if 'test' in st.session_state["study_results"]["splits"]:
            st.info("Study is finished. Save results or press plot button.")
        elif 'val' in st.session_state["study_results"]["splits"]:
            st.info("Study has been run for train and val data only. Press perform Button again to calculate for all data.")
        else:
            st.info("Study has been run for train data only. Press perform Button again to calculate for train and val data.")
        # Rerun
        st.rerun()

    if st.session_state.get("study_results", None) is not None:
        st.session_state["study_results"] = unpack_study_results(st.session_state.get("study_results_pack", None))
        if 'test' in st.session_state["study_results"]["splits"]:
            st.info("Study is finished. Save results or press plot button.")
        elif 'val' in st.session_state["study_results"]["splits"]:
            st.info("Study has been run for train and val data only. Press perform Button again to calculate for all data.")
        else:
            st.info("Study has been run for train data only. Press perform Button again to calculate for train and val data.")

    results = st.session_state.get("study_results", None)
    if results is None:
        try:
            results = _load_study_results_from_session()
        except:
            with header_slot.container():
                theme_header("Evaluate study", key="evaluate_study", right_extra=_header_export_button)
            st.info("No study results in session. Please run a study first on 'Perform evaluation study'.")
            return
        
    study_has_results = (
    isinstance(results, dict)
    and "splits" in results
    )
        
    if study_has_results: 
        # Define the plot kinds you support on this page + friendly labels
        plot_options: list[tuple[PlotKind, str]] = [
            (PlotKind.FEATURES_HIST, "Feature histograms"),
            (PlotKind.FC_EVAL_HIST, "Forecast-eval histograms"),
            (PlotKind.PRED_HIST, "Prediction histograms"),
            (PlotKind.VIOLIN_PRED, "Prediction violins (mean/lower/upper)"),
            (PlotKind.SCATTER_FEATURES, "Scatter of Features"),
            (PlotKind.SCATTER_CLUSTER, "Clustering assignment"),
            (PlotKind.HIST_CLUSTER, "Histogram of clusters"),
            (PlotKind.LINE_FEATURES, "Line plots of octant time series"),
            (PlotKind.PRED_EVAL_HIST, "Prediction-eval histograms"),
            (PlotKind.ASSOC_HEATMAPS, "Association heatmaps (ρ in [-1, 1])"),
            (PlotKind.BEST_METHOD_DIST, "Best method frequency"),
            (PlotKind.BEST_METHOD_QUADRANT, "Best-method evaluation"),
            (PlotKind.VIOLIN_PRED_EVAL, "Prediction-eval violins"),
        ]

        st.markdown("---")
        st.subheader("Plots to generate")
        b1, b2 = st.columns(2)

        if b1.button("Select all", key="study_plot_select_all"):
            for kind, _ in plot_options:
                st.session_state[f"chk_plot_{kind.value}"] = True
            st.rerun()

        if b2.button("Clear", key="study_plot_clear"):
            for kind, _ in plot_options:
                st.session_state[f"chk_plot_{kind.value}"] = False
            st.rerun()

        with st.form("plot_picker_form", clear_on_submit=False):
            cols = st.columns(2, gap="large")

            # Build selection purely from checkbox widget values
            selected_now: set[PlotKind] = set()
            for i, (kind, label) in enumerate(plot_options):
                col = cols[i % 2]
                if col.checkbox(label, key=f"chk_plot_{kind.value}"):
                    selected_now.add(kind)

            # Submit button: only now we store selection + optionally build missing plots
            submitted = st.form_submit_button("Plot selected")

        if submitted:
            st.session_state["study_plot_kinds_selected"] = selected_now

            already_built: set[PlotKind] = set(st.session_state.get("study_plot_kinds_built", set()))
            missing = [k for k in selected_now if k not in already_built]

            if missing:
                tmp = st.empty()  # everything rendered while plotting goes in here and can be cleared
                with tmp.container():
                    with st.spinner("Building selected plots ..."):
                        new_collector = build_all(results, kinds=missing, splits=("train", "val", "test"))

                        _ = apply_pretty_names_to_collector(new_collector, case_insensitive=True, whole_words=False)

                        collector: PlotCollector | None = st.session_state.get("plot_collector")
                        if collector is None:
                            collector = new_collector
                        else:
                            # merge new artifacts into existing collector (avoid rendering return values)
                            if hasattr(collector, "add"):
                                for art in new_collector.all():
                                    _ = collector.add(art)
                            else:
                                # fallback: replace if no merge API exists
                                collector = new_collector

                        st.session_state["plot_collector"] = collector
                        st.session_state["study_plot_kinds_built"] = already_built.union(set(missing))

                tmp.empty()

            st.rerun()


    if st.session_state.get("study_results", None) is not None:
        # Render header in reserved slot - button active
        with header_slot.container():
            theme_header(
                "Evaluate study",
                key="evaluate_study",
                right_extra=_header_export_button
            )

        collector: PlotCollector | None = st.session_state.get("plot_collector")

        if collector and collector.all():
            # Which plot kinds actually exist in the collector?
            available_kinds = []
            for kv in sorted({a.kind for a in collector.all()}):
                try:
                    available_kinds.append(PlotKind(kv))
                except Exception:
                    pass

            st.markdown("---")
            st.subheader("Displayed plots")

            display_selected = st.multiselect(
                "Filter plot types",
                options=available_kinds,
                default=[],
                format_func=lambda k: k.value,
                key="study_plot_display_filter",
            )

            # Semantics:
            # empty selection -> show ALL available plots
            display_kinds = (
                available_kinds
                if len(display_selected) == 0
                else display_selected
            )

            _show_study_results(results, display_kinds)
        else:
            st.info("No plots available yet. Select plot types and click 'Plot selected'.")

_page()