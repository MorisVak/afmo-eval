
import io
import re as _re_mod
import streamlit as st
from streamlit.delta_generator import DeltaGenerator as _DG

def _sanitize(s: str) -> str:
    s = (s or "").strip()
    s = _re_mod.sub(r"[^A-Za-z0-9_. -]+", "_", s)
    s = _re_mod.sub(r"\s+", "_", s)
    return s[:120] if len(s) > 120 else s

def install_capture_hooks() -> None:
    """Capture for Matplotlib + Plotly figures."""
    # reset per-run lists; keep built ZIP between runs until rebuilt
    st.session_state["__all_figs__"] = []
    st.session_state["__all_figs_labels__"] = []
    st.session_state["__all_figs_kinds__"] = []
    st.session_state.setdefault("__all_plots_zip_bytes", None)
    st.session_state.setdefault("__all_figs_count__", 0)

    def _infer_kind_and_label(obj):
        kind, label = "plot", None
        # Plotly?
        try:
            import plotly.graph_objects as _go
            if isinstance(obj, _go.Figure):
                try:
                    label = (obj.layout.title.text or "").strip() if obj.layout and obj.layout.title else None
                except Exception:
                    label = None
                txt = (label or "").lower()
                try:
                    ttypes = {getattr(t, "type", "") for t in obj.data}
                except Exception:
                    ttypes = set()
                if "heatmap" in ttypes or "heatmap" in txt:
                    kind = "heatmap"
                elif "violin" in ttypes or "violin" in txt:
                    kind = "violin"
                elif "histogram" in ttypes or "hist" in txt or "histogram" in txt:
                    kind = "hist"
                return kind, label
        except Exception:
            pass
        # Matplotlib?
        try:
            import matplotlib.figure as _mpl
            if isinstance(obj, _mpl.Figure):
                # read label/suptitle if set
                try:
                    label = getattr(obj, "_afmo_title", None) or (obj._suptitle.get_text().strip() if getattr(obj, "_suptitle", None) else None)
                except Exception:
                    label = None
                txt = (label or "").lower()
                if "heatmap" in txt or "spearman" in txt or "correlation" in txt:
                    kind = "heatmap"
                elif "violin" in txt:
                    kind = "violin"
                elif "hist" in txt or "histogram" in txt:
                    kind = "hist"
                elif "scatter" in txt:
                    kind = "scatter"
                else:
                    # heuristic: check axes artists for violin keywords
                    try:
                        for ax in obj.get_axes():
                            for art in getattr(ax, "collections", []):
                                n = art.__class__.__name__.lower()
                                if "violin" in n:
                                    return "violin", label
                    except Exception:
                        pass
                return kind, label
        except Exception:
            pass
        return kind, label

    st.session_state["__infer_kind_and_label__"] = _infer_kind_and_label
    st.session_state["__sanitize__"] = _sanitize

    # Patch pyplot once
    if not getattr(_DG.pyplot, "__afmo_patched__", False):
        _orig_pyplot = _DG.pyplot
        def _capturing_pyplot(self, *args, **kwargs):
            fig = args[0] if args else kwargs.get("figure") or kwargs.get("fig")
            if fig is not None:
                kind, label = st.session_state["__infer_kind_and_label__"](fig)
                st.session_state["__all_figs__"].append(fig)
                st.session_state["__all_figs_labels__"].append(label)
                st.session_state["__all_figs_kinds__"].append(kind)
            return _orig_pyplot(self, *args, **kwargs)
        _capturing_pyplot.__afmo_patched__ = True
        _DG.pyplot = _capturing_pyplot

    # Patch plotly_chart once
    if hasattr(_DG, "plotly_chart") and not getattr(_DG.plotly_chart, "__afmo_patched__", False):
        _orig_plotly_chart = _DG.plotly_chart
        def _capturing_plotly(self, figure_or_data=None, **kwargs):
            fig = figure_or_data
            if fig is not None:
                kind, label = st.session_state["__infer_kind_and_label__"](fig)
                st.session_state["__all_figs__"].append(fig)
                st.session_state["__all_figs_labels__"].append(label)
                st.session_state["__all_figs_kinds__"].append(kind)
            return _orig_plotly_chart(self, figure_or_data=figure_or_data, **kwargs)
        _capturing_plotly.__afmo_patched__ = True
        _DG.plotly_chart = _capturing_plotly

def render_top_download_button() -> None:
    """Render the one top button; disabled until ZIP is ready."""
    zip_bytes = st.session_state.get("__all_plots_zip_bytes")
    st.download_button(
        label="📦 Download all plots as .zip",
        data=zip_bytes if zip_bytes else b"",
        file_name="evaluation_plots.zip",
        mime="application/zip",
        disabled=(zip_bytes is None),
        help="Activated when plots are rendered."
    )

def build_zip_end() -> None:
    """Build the ZIP from captured figures."""
    try:
        import plotly.io as _pio
        import kaleido
        import zipfile
        figs = st.session_state.get("__all_figs__", []) or []
        labels = st.session_state.get("__all_figs_labels__", []) or []
        kinds = st.session_state.get("__all_figs_kinds__", []) or []
        _sanitize = st.session_state.get("__sanitize__", lambda x: x or "figure")
        if figs and len(figs) != st.session_state.get("__all_figs_count__", 0):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, fig in enumerate(figs):
                    base = labels[idx] if idx < len(labels) and labels[idx] else f"figure_{idx+1}"
                    kind = kinds[idx] if idx < len(kinds) and kinds[idx] else "plot"
                    base = _sanitize(base)
                    if not base.startswith(f"{kind}_"):
                        base = f"{kind}_{base}"
                    # Plotly: HTML always, PNG if kaleido
                    try:
                        import plotly.graph_objects as _go
                        if isinstance(fig, _go.Figure):
                            html = _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
                            zf.writestr(f"{base}.html", html.encode("utf-8"))
                            try:
                                png = fig.to_image(format="png", scale=2)
                                zf.writestr(f"{base}.png", png)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass
                    # Matplotlib: PNG + PDF
                    try:
                        import matplotlib.figure as _mpl
                        if isinstance(fig, _mpl.Figure):
                            png_buf = io.BytesIO()
                            fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
                            zf.writestr(f"{base}.png", png_buf.getvalue())
                            pdf_buf = io.BytesIO()
                            fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
                            zf.writestr(f"{base}.pdf", pdf_buf.getvalue())
                            continue
                    except Exception:
                        pass
            buf.seek(0)
            st.session_state["__all_plots_zip_bytes"] = buf.getvalue()
            st.session_state["__all_figs_count__"] = len(figs)
    except Exception:
        pass
