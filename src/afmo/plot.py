"""Utilities for AFMo (src/afmo/plot.py)"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

def history_plot(df: pd.DataFrame, columns: list) -> go.Figure:
    """
    Robust line plot for one or more series.
    - Only plots columns that exist and have at least 2 finite points
    - Coerces to numeric and drops NaNs
    - Downsamples if necessary to avoid rendering issues
    """
    fig = go.Figure()
    if df is None or df.empty or not columns:
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    # Ensure DatetimeIndex or an ordered numeric index
    x_index = df.index
    # Downsample index if very long to reduce canvas load
    max_points = 20000  # safe upper bound for smooth plotting
    step = max(1, int(len(x_index) / max_points))

    plotted = False
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.shape[0] < 2:
            continue
        s = s.iloc[::step]  # position-based indexing into DataFrame
        xi = s.index
        yi = s.values
        if len(yi) < 2:
            continue
        fig.add_scatter(x=xi, y=yi, mode="lines", name=str(col))
        plotted = True

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if not plotted:
        # Add a small annotation instead of returning an empty/invalid canvas
        fig.add_annotation(text="No plottable data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def forecast_plot(df: pd.DataFrame, base_col: str, fc: pd.DataFrame, title: str = "Forecast") -> go.Figure:
    # Plot history from `df[base_col]` plus a forecast DataFrame `fc`
    # with columns: mean[, lower, upper] and a DatetimeIndex.
    fig = go.Figure()
    if base_col in df.columns:
        fig.add_scatter(x=df.index, y=df[base_col], mode="lines", name=f"History: {base_col}")
    if "mean" in fc.columns:
        fig.add_scatter(x=fc.index, y=fc["mean"], mode="lines", name="Forecast")
    if {"lower", "upper"}.issubset(fc.columns):
        fig.add_scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["lower"]) + list(fc["upper"][::-1]),
            fill="toself",
            opacity=0.2,
            name="Interval",
            mode="lines",
            showlegend=True,
        )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def make_plotly_figures_nested(out: dict, splits=("train", "val", "test")) -> Dict[str, Dict[str, Any]]:
    """
    Build a nested dict of Plotly figures:
      figs[split]["features"][<feature>] = histogram
      figs[split]["fc_eval"][<metric>] = histogram
      figs[split]["pred"][<metric>] = histogram (means), colored by predictor
      figs[split]["pred_eval"][<pred_metric>][<metric>] = histogram, colored by predictor
      figs[split]["best_method"]["distribution"] = bar chart
      figs["test"]["best_method_eval"]["quadrant"] = scatter with thresholds
    """
    figs: Dict[str, Dict[str, Any]] = {}

    for split in splits:
        figs[split] = {}

        # features per feature
        try:
            df_feat = _flatten_features(out, split)
            if not df_feat.empty:
                figs[split]["features"] = {}
                for feature, d in df_feat.groupby("feature"):
                    fig = px.histogram(d, x="value", nbins=40, opacity=0.85,
                                       title=f"Feature histogram — {split} — feature: {feature}")
                    fig.update_layout(bargap=0.02, xaxis_title="value", yaxis_title="count")
                    figs[split]["features"][str(feature)] = fig
        except KeyError:
            pass

        # fc_eval per metric
        try:
            df_fce = _flatten_fc_eval(out, split)
            if not df_fce.empty:
                figs[split]["fc_eval"] = {}
                for metric, d in df_fce.groupby("metric"):
                    fig = px.histogram(d, x="value", nbins=40, opacity=0.85,
                                       title=f"fc_eval histogram — {split} — metric: {metric}")
                    fig.update_layout(bargap=0.02, xaxis_title="value", yaxis_title="count")
                    figs[split]["fc_eval"][str(metric)] = fig
        except KeyError:
            pass

        # pred per forecast metric, colored by predictor
        try:
            df_pred = _flatten_pred(out, split)
            if not df_pred.empty:
                figs[split]["pred"] = {}
                for metric, d in df_pred.groupby("metric"):
                    fig = px.histogram(d, x="value", color="predictor", nbins=40, opacity=0.75,
                                       title=f"pred histogram (mean) — {split} — metric: {metric}")
                    fig.update_layout(bargap=0.02, xaxis_title="mean value", yaxis_title="count")
                    figs[split]["pred"][str(metric)] = fig
        except KeyError:
            pass

        # pred_eval per pred_metric × metric, colored by predictor
        try:
            df_pe = _flatten_pred_eval(out, split)
            if not df_pe.empty:
                figs[split]["pred_eval"] = {}
                for (pred_metric, metric), d in df_pe.groupby(["pred_metric", "metric"]):
                    figs[split]["pred_eval"].setdefault(str(pred_metric), {})
                    fig = px.histogram(d, x="value", color="predictor", nbins=40, opacity=0.75,
                                       title=f"pred_eval histogram — {split} — pred_metric: {pred_metric} — metric: {metric}")
                    fig.update_layout(bargap=0.02, xaxis_title="value", yaxis_title="count")
                    figs[split]["pred_eval"][str(pred_metric)][str(metric)] = fig
        except KeyError:
            pass

        # best_method distribution
        try:
            df_bm = _flatten_best_method(out, split)
            if not df_bm.empty:
                counts = df_bm["best_method"].value_counts().reset_index()
                counts.columns = ["best_method", "count"]
                fig = px.bar(counts, x="best_method", y="count", text="count",
                             title=f"best_method frequency — {split}")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="best_method", yaxis_title="count")
                figs[split].setdefault("best_method", {})
                figs[split]["best_method"]["distribution"] = fig
        except KeyError:
            pass

    # quadrant only for test, if available
    try:
        quad = out["splits"]["test"]["best_method_eval"]["summary"]["quadrant"]
        x_vals: dict = quad["x_values"]
        y_vals: dict = quad["y_values"]
        correct: dict = quad["correct_values"]
        x_thr = quad.get("x_thresh_default", 0.5)
        y_thr = quad.get("y_thresh_default", None)

        rows = []
        for ts, x in x_vals.items():
            y = y_vals.get(ts)
            if y is None:
                continue
            c = bool(correct.get(ts, False))
            rows.append({"ts": ts, "x": float(x), "y": float(y), "correct": "correct" if c else "wrong"})
        dfq = pd.DataFrame(rows)
        if not dfq.empty:
            figq = px.scatter(dfq, x="x", y="y", color="correct", hover_name="ts",
                              title="Quadrant best_method_eval — test",
                              labels={"x": quad.get("x_metric", "x"), "y": quad.get("y_metric", "y")})
            figq.add_shape(type="line", x0=x_thr, x1=x_thr, y0=dfq["y"].min(), y1=dfq["y"].max(), line=dict(dash="dash"))
            if y_thr is not None:
                figq.add_shape(type="line", x0=dfq["x"].min(), x1=dfq["x"].max(), y0=y_thr, y1=y_thr, line=dict(dash="dash"))
            figq.update_layout(xaxis_title=f"{quad.get('x_metric','x')} (higher = earlier match)",
                               yaxis_title=f"{quad.get('y_metric','y')} (higher = more confidence)")
            figs.setdefault("test", {}).setdefault("best_method_eval", {})
            figs["test"]["best_method_eval"]["quadrant"] = figq
    except KeyError:
        pass

    return figs