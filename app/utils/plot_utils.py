
"""
plot_utils.py — figure builder/renderer for Evaluation page histograms.
"""
from __future__ import annotations
from typing import Any
import streamlit as st

# legacy nested builder
_legacy_make_figs = None
_legacy_err = None
from afmo.plot import make_plotly_figures_nested as _legacy_make_figs  # type: ignore

# grouped histogram builders
_make_hists = None
_hists_err = None
from afmo.plot import make_histograms_for_split as _make_hists  # type: ignore


def build_figs(split: dict) -> dict:
    """Return a nested mapping of figures; fall back across backends gracefully."""
    # Attempt legacy first
    if _legacy_make_figs is not None:
        try:
            figs = _legacy_make_figs(split or {})
            if isinstance(figs, dict) and figs:
                return figs
        except Exception as e:
            st.warning(f"build_figs: legacy builder failed; using histogram fallback. ({e})")

    # Fallback: grouped histograms -> convert to legacy shape
    if _make_hists is None:
        st.error(f"build_figs: no figure builder available.")
        return {}

    groups = _make_hists(split or {})
    out = {"features": {}, "fc_eval": {}, "pred": {}, "pred_eval": {}, "best_method": {}}
    for i, fig in enumerate(groups.get("features", [])):
        out["features"][f"feature_{i+1}"] = fig
    for i, fig in enumerate(groups.get("fc_eval", [])):
        out["fc_eval"][f"metric_{i+1}"] = fig
    for i, fig in enumerate(groups.get("pred", [])):
        out["pred"][f"metric_{i+1}"] = fig
    inner = {}
    for i, fig in enumerate(groups.get("pred_eval", [])):
        inner[f"combo_{i+1}"] = fig
    if inner:
        out["pred_eval"]["all"] = inner
    for i, fig in enumerate(groups.get("best_method", [])):
        out["best_method"][f"dist_{i+1}"] = fig
    return out


def _is_plotly_fig(obj: Any) -> bool:
    try:
        return hasattr(obj, "to_plotly_json")
    except Exception:
        return False


def _looks_like_figs_dict(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    keys = set(d.keys())
    expected = {"features", "fc_eval", "pred", "pred_eval", "best_method", "best_method_eval"}
    if keys & expected:
        return True
    for v in d.values():
        if _is_plotly_fig(v):
            return True
        if isinstance(v, dict) and any(_is_plotly_fig(x) for x in v.values()):
            return True
    return False


def _render_figs_dict(col, figs: dict):
    order = ["features", "fc_eval", "pred_eval", "pred", "best_method", "best_method_eval"]
    for section in order:
        bucket = figs.get(section, {})
        if not bucket:
            continue
        if section == "pred_eval" and any(isinstance(v, dict) for v in bucket.values()):
            col.markdown("**Prediction evaluation (pred_eval) — histograms (by predictor)**")
            for outer_key in sorted(bucket.keys(), key=lambda x: str(x)):
                inner = bucket[outer_key] or {}
                for inner_key in sorted(inner.keys(), key=lambda x: str(x)):
                    col.plotly_chart(inner[inner_key], use_container_width=True)
        else:
            title_map = {
                "features": "**Features — histograms**",
                "fc_eval": "**Forecast metrics (fc_eval) — histograms**",
                "pred": "**Prediction means (pred) — histograms (by predictor)**",
                "best_method": "**Best method — distribution**",
                "best_method_eval": "**Best method (eval) — distribution**",
            }
            if section in title_map:
                col.markdown(title_map[section])
            for key in sorted(bucket.keys(), key=lambda x: str(x)):
                col.plotly_chart(bucket[key], use_container_width=True)


def render_split(col, *args, **kwargs):
    """
    Render either a split dict (build histograms on the fly) or a figs dict (prebuilt).
    """
    given = None
    for a in args[1:]:
        if isinstance(a, dict):
            given = a
            break
    if given is None:
        given = kwargs.get("split", None) or kwargs.get("figs", None)

    if not isinstance(given, dict):
        col.caption("No data to render."); 
        return

    if _looks_like_figs_dict(given):
        _render_figs_dict(col, given)
        return

    if _make_hists is None:
        col.error(f"No histogram backend available. {_hists_err}")
        return

    groups = _make_hists(given)
    if groups.get("features"):
        col.markdown("**Features — histograms**")
        for fig in groups["features"]:
            col.plotly_chart(fig, use_container_width=True)
    if groups.get("fc_eval"):
        col.markdown("**Forecast metrics (fc_eval) — histograms**")
        for fig in groups["fc_eval"]:
            col.plotly_chart(fig, use_container_width=True)
    if groups.get("pred_eval"):
        col.markdown("**Prediction evaluation (pred_eval) — histograms (by predictor)**")
        for fig in groups["pred_eval"]:
            col.plotly_chart(fig, use_container_width=True)
    if groups.get("pred"):
        col.markdown("**Prediction means (pred) — histograms (by predictor)**")
        for fig in groups["pred"]:
            col.plotly_chart(fig, use_container_width=True)
    if groups.get("best_method"):
        col.markdown("**Best method — distribution**")
        for fig in groups["best_method"]:
            col.plotly_chart(fig, use_container_width=True)
