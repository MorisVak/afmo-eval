from __future__ import annotations
from typing import Iterable
from .base import PlotArtifact, Split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _mk_fig(M, rows, cols, title: str, xlabel: str, ylabel: str):
    M = np.asarray(M, dtype=float) if M is not None else np.empty((0, 0))
    if M.size == 0:
        return None
    h, w = M.shape
    fig = plt.figure(figsize=(min(16, 1.25*(2.5 + 0.35*w)), min(12, 2.5 + 0.35*h)), facecolor="white")
    ax = fig.gca(); ax.set_facecolor("white")
    im = ax.imshow(M, aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(list(range(len(cols))))
    ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([str(r) for r in rows])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("ρ")
    fig.tight_layout()
    return fig

def assoc(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Build association heatmaps (Spearman & Partial Spearman) directly from the
    association dict produced by study.run_experiment (key: "assoc" or "associations").
    Returns PlotArtifacts with Matplotlib figures.

    Original families (for "spearman"):
      - fc_eval vs. features              (from "features_vs_fc_eval")
      - pred_vs_features                  (from "pred_eval_vs_features") — one heatmap per (predictor × fc_metric)
      - pred_vs_fc                        (from "pred_eval_vs_fc_eval") — one heatmap per predictor
      - predictor_vs_features             (derived from "pred_eval_vs_features" by aggregating across fc_metrics)
      - predictor_vs_fc_eval              (pivoted from "pred_eval_vs_fc_eval")

    Additional aggregated families (raw values aggregated first, then Spearman):
      - pred_vs_features__agg_fc          from ...[predictor][feature][pred_metric]["__agg_fc__"]
      - features_vs_pred_eval__agg_pred   from ...[feature][pred_metric]["__agg_pred__"][fc_metric]
      - features_vs_pred_eval__agg_fc_pred from ...[feature][pred_metric]["__agg_fc_pred__"]
      - pred_eval_vs_fc_eval__agg_pred    from ...["pred_eval_vs_fc_eval"]["__agg_pred__"][pred_metric][fc_metric]
    """
    import numpy as np
    import matplotlib.patheffects as pe

    from afmo.core.registry import PREDICTORS, FC_METRICS_SCORES, FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS
    predictors = list(PREDICTORS.keys())
    predictors.remove('best_method')
    fc_scores = list(FC_METRICS_SCORES.keys())
    fc_metric_pred = list(FC_METRICS_PREDICTABILITY.keys())
    fc_metric_eff = list(FC_METRICS_EFFECTIVENESS)
    fc_metrics = [fc_scores[0]] + fc_metric_pred + [fc_scores[1]] + fc_metric_eff
    fc_method = out['config']['model_name']

    # Soft caps for heatmap raster size to avoid OOM during PNG render
    MAX_ROWS, MAX_COLS = 400, 300
    # Soft cap for text annotations to avoid thousands of Text artists on huge matrices
    MAX_ANNOTATION_CELLS = 25000

    def _mk_fig_slim(M, rows, cols, title, xlabel="", ylabel="", P=None):
        """Downsample big matrices + lower DPI via _mk_fig. Then annotate cells with rho (1 dec) and '*' if p<.05."""
        M = np.asarray(M, dtype=np.float32)
        P = None if P is None else np.asarray(P, dtype=np.float32)

        r_idx = np.arange(len(rows))
        c_idx = np.arange(len(cols))
        if M.shape[0] > MAX_ROWS:
            r_idx = np.linspace(0, M.shape[0] - 1, MAX_ROWS).astype(int)
            rows = [rows[i] for i in r_idx]
            M = M[r_idx, :]
            if P is not None:
                P = P[r_idx, :]
        if M.shape[1] > MAX_COLS:
            c_idx = np.linspace(0, M.shape[1] - 1, MAX_COLS).astype(int)
            cols = [cols[j] for j in c_idx]
            M = M[:, c_idx]
            if P is not None:
                P = P[:, c_idx]

        fig = _mk_fig(M, rows, cols, title, xlabel=xlabel, ylabel=ylabel)
        if fig is None:
            return None
        try:
            fig.set_dpi(min(fig.get_dpi(), 90))
        except Exception:
            pass

        # Annotate rho and significance star
        try:
            ax = fig.axes[0] if fig.axes else fig.gca()
            nrows, ncols = M.shape
            total = nrows * ncols
            stride = 1
            if total > MAX_ANNOTATION_CELLS:
                stride = int(np.ceil(np.sqrt(total / MAX_ANNOTATION_CELLS)))
                stride = max(1, stride)

            # Adaptive font size: smaller for larger grids
            base_fs = 10
            scale = max(nrows, ncols)
            fs = max(5, min(base_fs, int(base_fs - 0.012 * scale)))

            for i in range(0, nrows, stride):
                for j in range(0, ncols, stride):
                    rho = M[i, j]
                    if not np.isfinite(rho):
                        continue
                    pval = np.nan
                    if P is not None and i < P.shape[0] and j < P.shape[1]:
                        pval = P[i, j]
                    star = "*" if (np.isfinite(pval) and pval < 0.05) else ""
                    s = f"{rho:.1f}{star}"
                    ax.text(
                        j, i, s,
                        ha="center", va="center",
                        fontsize=fs, color="black",
                        path_effects=[pe.withStroke(linewidth=0.5, foreground="white")]
                    )
        except Exception:
            # Do not fail plotting if annotation crashes for any reason
            pass
        return fig

    arts: list[PlotArtifact] = []
    split_obj = (out or {}).get("splits", {}).get(split) or {}
    assoc = (split_obj.get("assoc") or split_obj.get("associations") or {})
    if not assoc:
        return arts

    def _known_features(block) -> set[str]:
        feats = set()
        fblock = (block or {}).get("features_vs_fc_eval") or {}
        for _fc, payload in (fblock or {}).items():
            feats.update((((payload or {}).get("features")) or {}).keys())
        return feats

    def fc_eval_vs_features(block, label: str):
        fc_block = (block or {}).get("features_vs_fc_eval") or {}
        if not fc_block:
            return
        # fc_metrics = sorted(fc_block.keys(), key=str)
        feats = sorted(
            {feat for fc in fc_metrics for feat in (((fc_block.get(fc) or {}).get("features")) or {}).keys()},
            key=str,
        )
        if not feats or not fc_metrics:
            return
        M = np.full((len(feats), len(fc_metrics)), np.nan, dtype=np.float32)
        P = np.full_like(M, np.nan, dtype=np.float32)
        for j, fc in enumerate(fc_metrics):
            feats_map = ((fc_block.get(fc) or {}).get("features")) or {}
            for i, feat in enumerate(feats):
                res = (feats_map.get(feat) or {})
                rho = res.get("rho")
                p = res.get("pvalue")
                M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
        title = f"{label}: Forecast accuracy vs. features ({fc_method})"
        fig = _mk_fig_slim(M, feats, fc_metrics, title, xlabel="FC metric", ylabel="Feature", P=P)
        if fig is not None:
            arts.append(PlotArtifact(
                id=f"hm_{label.lower().replace(' ', '_')}_fc_feat",
                title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
            ))

    def pred_vs_features(block, label: str):
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        feats_known = _known_features(block)
        predictor_keys = [k for k in (pvf or {}).keys()
                          if not str(k).startswith("__") and k not in feats_known]
        for predictor in predictors: # sorted(predictor_keys, key=str)
            feat_map = pvf.get(predictor) or {}
            feats = sorted((feat_map or {}).keys(), key=str)
            pred_metrics = sorted({pm for feat in feats for pm in ((feat_map.get(feat) or {}).keys())}, key=str)
            fc_metrics = sorted({
                fc for feat in feats
                for pm in ((feat_map.get(feat) or {}).keys())
                for fc in (((feat_map.get(feat) or {}).get(pm) or {}).keys())
                if not str(fc).startswith("__")
            }, key=str)
            for fc in fc_metrics:
                if not feats or not pred_metrics:
                    continue
                M = np.full((len(feats), len(pred_metrics)), np.nan, dtype=np.float32)
                P = np.full_like(M, np.nan, dtype=np.float32)
                for i, feat in enumerate(feats):
                    for j, pm in enumerate(pred_metrics):
                        res = (((feat_map.get(feat) or {}).get(pm) or {}).get(fc) or {})
                        rho = res.get("rho")
                        p = res.get("pvalue")
                        M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                        P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
                title = f"{label}: Forecast accuracy vs. features: {predictor} — {fc} ({fc_method})"
                fig = _mk_fig_slim(M, feats, pred_metrics, title, xlabel="Prediction metric", ylabel="Feature", P=P)
                if fig is not None:
                    arts.append(PlotArtifact(
                        id=f"hm_{label.lower().replace(' ', '_')}_pvf_{predictor}_{fc}".replace(" ", "_"),
                        title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                    ))

    def pred_vs_fc(block, label: str):
        pvfc = (block or {}).get("pred_eval_vs_fc_eval") or {}
        for predictor in predictors: # sorted([k for k in (pvfc or {}).keys() if not str(k).startswith("__")], key=str):
            pm_map = pvfc.get(predictor) or {}
            pred_metrics = sorted((pm_map or {}).keys(), key=str)
            #fc_metrics = sorted({fc for pm in pred_metrics for fc in ((pm_map.get(pm) or {}).keys())}, key=str)
            if not pred_metrics or not fc_metrics:
                continue
            M = np.full((len(pred_metrics), len(fc_metrics)), np.nan, dtype=np.float32)
            P = np.full_like(M, np.nan, dtype=np.float32)
            for i, pm in enumerate(pred_metrics):
                for j, fc in enumerate(fc_metrics):
                    res = ((pm_map.get(pm) or {}).get(fc) or {})
                    rho = res.get("rho")
                    p = res.get("pvalue")
                    M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                    P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
            title = f"{label}: Prediction vs. forecast accuracy — {predictor} ({fc_method})"
            fig = _mk_fig_slim(M, pred_metrics, fc_metrics, title, xlabel="FC metric", ylabel="Prediction metric", P=P)
            if fig is not None:
                arts.append(PlotArtifact(
                    id=f"hm_{label.lower().replace(' ', '_')}_pvfc_{predictor}".replace(" ", "_"),
                    title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                ))

    def predictor_vs_features_from_pred_eval(block, label: str):
        # Legacy derived variant uses mean of rhos across fc; p-values are not defined here
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        if not pvf:
            return
        feats_known = _known_features(block)
        # predictors = sorted([k for k in (pvf or {}).keys()
        #                      if not str(k).startswith("__") and k not in feats_known], key=str)
        if not predictors:
            return
        feats = sorted({feat for pred in predictors for feat in ((pvf.get(pred) or {}).keys())}, key=str)
        pred_metrics_all = sorted({
            pm for pred in predictors
            for feat, pm_map in ((pvf.get(pred) or {}).items())
            for pm in (pm_map or {}).keys()
        }, key=str)
        for pm in pred_metrics_all:
            # fc_metrics = sorted({
            #     fc for pred in predictors
            #     for feat, pm_map in ((pvf.get(pred) or {}).items())
            #     for fc in (((pm_map.get(pm) or {}) or {}).keys())
            #     if not str(fc).startswith("__")
            # }, key=str)
            if not feats or not predictors or not fc_metrics:
                continue
            for fc in fc_metrics:
                M = np.full((len(feats), len(predictors)), np.nan, dtype=np.float32)
                P = np.full_like(M, np.nan, dtype=np.float32)
                for j, predictor in enumerate(predictors):
                    feat_map = (pvf.get(predictor) or {})
                    for i, feat in enumerate(feats):
                        res_map = ((feat_map.get(feat) or {}).get(pm) or {})
                        rho = (res_map.get(fc) or {}).get("rho")
                        p = (res_map.get(fc) or {}).get("pvalue")
                        M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                        P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
                title = f"{label}: Estimator vs. features — {pm} - {fc} ({fc_method})"
                fig = _mk_fig_slim(M, feats, predictors, title, xlabel="Predictor", ylabel="Feature", P=P)
                if fig is not None:
                    arts.append(PlotArtifact(
                        id=f"hm_{label.lower().replace(' ', '_')}_predictor_features_{pm}_{fc}".replace(" ", "_"),
                        title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                    ))

    def predictor_vs_fc_eval_from_pred_eval(block, label: str):
        pvfc = (block or {}).get("pred_eval_vs_fc_eval") or {}
        if not pvfc:
            return
        #predictors = sorted([k for k in (pvfc or {}).keys() if not str(k).startswith("__")], key=str)
        pred_metrics_all = sorted({
            pm for pred in predictors
            for pm in ((pvfc.get(pred) or {}).keys())
        }, key=str)
        for pm in pred_metrics_all:
            # fc_metrics = sorted({
            #     fc for pred in predictors
            #     for fc in (((pvfc.get(pred) or {}).get(pm) or {}).keys())
            # }, key=str)
            if not predictors or not fc_metrics:
                continue
            M = np.full((len(fc_metrics), len(predictors)), np.nan, dtype=np.float32)
            P = np.full_like(M, np.nan, dtype=np.float32)
            for j, predictor in enumerate(predictors):
                pm_map = ((pvfc.get(predictor) or {}).get(pm) or {})
                for i, fc in enumerate(fc_metrics):
                    res = (pm_map.get(fc) or {})
                    rho = res.get("rho")
                    p = res.get("pvalue")
                    M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                    P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
            title = f"{label}: Estimator vs. forecast accuracy — {pm} ({fc_method})"
            fig = _mk_fig_slim(M, fc_metrics, predictors, title, xlabel="Predictor", ylabel="FC metric", P=P)
            if fig is not None:
                arts.append(PlotArtifact(
                    id=f"hm_{label.lower().replace(' ', '_')}_predictor_fc_{pm}".replace(" ", "_"),
                    title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                ))

    # --- aggregated-variant heatmaps ---
    def pred_vs_features__agg_fc(block, label: str):
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        if not pvf:
            return
        feats_known = _known_features(block)
        predictor_keys = [k for k in (pvf or {}).keys()
                          if not str(k).startswith("__") and k not in feats_known]
        for predictor in predictors: # sorted(predictor_keys, key=str):
            feat_map = pvf.get(predictor) or {}
            feats = sorted((feat_map or {}).keys(), key=str)
            pred_metrics = sorted({pm for feat in feats for pm in ((feat_map.get(feat) or {}).keys())}, key=str)
            if not feats or not pred_metrics:
                continue
            M = np.full((len(feats), len(pred_metrics)), np.nan, dtype=np.float32)
            P = np.full_like(M, np.nan, dtype=np.float32)
            for i, feat in enumerate(feats):
                for j, pm in enumerate(pred_metrics):
                    res = (((feat_map.get(feat) or {}).get(pm) or {}).get("__agg_fc__") or {})
                    rho = res.get("rho")
                    p = res.get("pvalue")
                    M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                    P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
            title = f"{label}: Estimation accuracy vs. features — {predictor} ({fc_method})"
            fig = _mk_fig_slim(M, feats, pred_metrics, title, xlabel="Prediction metric", ylabel="Feature", P=P)
            if fig is not None:
                arts.append(PlotArtifact(
                    id=f"hm_{label.lower().replace(' ', '_')}_pvf_aggfc_{predictor}".replace(" ", "_"),
                    title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                ))

    def features_vs_pred_eval__agg_pred(block, label: str):
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        if not pvf:
            return
        feats_known = _known_features(block)
        feats = sorted([k for k in (pvf or {}).keys() if k in feats_known], key=str)
        if not feats:
            return
        pred_metrics = sorted({
            pm for feat in feats
            for pm in ((pvf.get(feat) or {}).keys())
            if "__agg_pred__" in ((pvf.get(feat) or {}).get(pm) or {})
        }, key=str)
        # fc_metrics = sorted({
        #     fc for feat in feats
        #     for pm in pred_metrics
        #     for fc in ((((pvf.get(feat) or {}).get(pm) or {}).get("__agg_pred__") or {}).keys())
        # }, key=str)
        for fc in fc_metrics:
            M = np.full((len(feats), len(pred_metrics)), np.nan, dtype=np.float32)
            P = np.full_like(M, np.nan, dtype=np.float32)
            for i, feat in enumerate(feats):
                pm_map = pvf.get(feat) or {}
                for j, pm in enumerate(pred_metrics):
                    res = (((pm_map.get(pm) or {}).get("__agg_pred__") or {}).get(fc) or {})
                    rho = res.get("rho")
                    p = res.get("pvalue")
                    M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                    P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
            title = f"{label}: Features vs estimation accuracy — {fc} ({fc_method})"
            fig = _mk_fig_slim(M, feats, pred_metrics, title, xlabel="Prediction metric", ylabel="Feature", P=P)
            if fig is not None:
                arts.append(PlotArtifact(
                    id=f"hm_{label.lower().replace(' ', '_')}_feat_pred_aggpred_{fc}".replace(" ", "_"),
                    title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                ))

    def features_vs_pred_eval__agg_fc_pred(block, label: str):
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        if not pvf:
            return
        feats_known = _known_features(block)
        feats = sorted([k for k in (pvf or {}).keys() if k in feats_known], key=str)
        if not feats:
            return
        pred_metrics = sorted({
            pm for feat in feats
            for pm in ((pvf.get(feat) or {}).keys())
            if "__agg_fc_pred__" in ((pvf.get(feat) or {}).get(pm) or {})
        }, key=str)
        if not pred_metrics:
            return
        M = np.full((len(feats), len(pred_metrics)), np.nan, dtype=np.float32)
        P = np.full_like(M, np.nan, dtype=np.float32)
        for i, feat in enumerate(feats):
            pm_map = pvf.get(feat) or {}
            for j, pm in enumerate(pred_metrics):
                res = ((pm_map.get(pm) or {}).get("__agg_fc_pred__") or {})
                rho = res.get("rho")
                p = res.get("pvalue")
                M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
        title = f"{label}: Features vs. estimation accuracy ({fc_method})"
        fig = _mk_fig_slim(M, feats, pred_metrics, title, xlabel="Prediction metric", ylabel="Feature", P=P)
        if fig is not None:
            arts.append(PlotArtifact(
                id=f"hm_{label.lower().replace(' ', '_')}_feat_pred_aggfcpred",
                title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
            ))

    def pred_eval_vs_fc_eval__agg_pred(block, label: str):
        pvfc = (block or {}).get("pred_eval_vs_fc_eval") or {}
        agg = (pvfc or {}).get("__agg_pred__") or {}
        if not agg:
            return
        pred_metrics = sorted(agg.keys(), key=str)
        fc_metrics = sorted({fc for pm in pred_metrics for fc in ((agg.get(pm) or {}).keys())}, key=str)
        if not pred_metrics or not fc_metrics:
            return
        M = np.full((len(pred_metrics), len(fc_metrics)), np.nan, dtype=np.float32)
        P = np.full_like(M, np.nan, dtype=np.float32)
        for i, pm in enumerate(pred_metrics):
            for j, fc in enumerate(fc_metrics):
                res = ((agg.get(pm) or {}).get(fc) or {})
                rho = res.get("rho")
                p = res.get("pvalue")
                M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan
        title = f"{label}: estimation accuracy vs. forecast accuracy ({fc_method})"
        fig = _mk_fig_slim(M, pred_metrics, fc_metrics, title, xlabel="FC metric", ylabel="Prediction metric", P=P)
        if fig is not None:
            arts.append(PlotArtifact(
                id=f"hm_{label.lower().replace(' ', '_')}_pvfc_aggpred",
                title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
            ))

    # predictor_vs_features__agg_fc (reads from ...["pred_eval_vs_features"]["agg__fc"])
    def predictor_vs_features__agg_fc(block, label: str):
        pvf = (block or {}).get("pred_eval_vs_features") or {}
        if not pvf:
            return

        feats_known = _known_features(block)
        # predictors = sorted([k for k in pvf.keys() if not str(k).startswith("__") and k not in feats_known], key=str)
        if not predictors:
            return

        feats = sorted({feat
                        for pred in predictors
                        for feat, pm_map in (pvf.get(pred) or {}).items()
                        for pm, val in (pm_map or {}).items()
                        if isinstance(val, dict) and "__agg_fc__" in val}, key=str)

        pred_metrics_all = sorted({pm
                                for pred in predictors
                                for feat, pm_map in (pvf.get(pred) or {}).items()
                                for pm, val in (pm_map or {}).items()
                                if isinstance(val, dict) and "__agg_fc__" in val}, key=str)

        if not feats or not pred_metrics_all:
            return

        for pm in pred_metrics_all:
            M = np.full((len(feats), len(predictors)), np.nan, dtype=np.float32)
            P = np.full_like(M, np.nan, dtype=np.float32)
            for j, predictor in enumerate(predictors):
                feat_map = (pvf.get(predictor) or {})
                for i, feat in enumerate(feats):
                    res = (((feat_map.get(feat) or {}).get(pm) or {}).get("__agg_fc__") or {})
                    rho = res.get("rho")
                    p = res.get("pvalue")
                    M[i, j] = float(rho) if isinstance(rho, (int, float)) else np.nan
                    P[i, j] = float(p) if isinstance(p, (int, float)) else np.nan

            title = f"{label}: Estimator vs. features — {pm} ({fc_method})"
            fig = _mk_fig_slim(M, feats, predictors, title, xlabel="Predictor", ylabel="Feature", P=P)
            if fig is not None:
                arts.append(PlotArtifact(
                    id=f"hm_{label.lower().replace(' ', '_')}_predictor_features_aggfc_{pm}".replace(" ", "_"),
                    title=title, kind="heatmap/assoc", split=split, fig=fig, lib="mpl"
                ))


    # --- Spearman plots ---
    sp = assoc.get("spearman") or {}
    # Baselines
    fc_eval_vs_features(sp, "Spearman")
    pred_vs_features(sp, "Spearman")
    pred_vs_fc(sp, "Spearman")
    predictor_vs_features_from_pred_eval(sp, "Spearman")
    predictor_vs_fc_eval_from_pred_eval(sp, "Spearman")
    # aggregated variants
    pred_vs_features__agg_fc(sp, "Spearman")
    features_vs_pred_eval__agg_pred(sp, "Spearman")
    features_vs_pred_eval__agg_fc_pred(sp, "Spearman")
    pred_eval_vs_fc_eval__agg_pred(sp, "Spearman")
    predictor_vs_features__agg_fc(sp, "Spearman")

    return arts

def best_worst_feature_bars(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Wrap the existing 'best/worst features' bar figures produced by afmo.plot.
    """
    arts: list[PlotArtifact] = []
    try:
        from importlib import import_module
        _plot = import_module("afmo.plot")
        for title, fig in _plot.make_pred_feature_heatmaps_for_split(out, split):
            arts.append(PlotArtifact(
                id=f"predfeat_{title}".replace(" ", "_"),
                title=title, kind="bar/best_worst_features", split=split, fig=fig, lib="mpl"
            ))
    except Exception:
        pass
    return arts

def make_pred_eval_violins_for_split_agg(dict_res: dict, split_name: str):
    """
    Build Seaborn/Matplotlib violin figures (only body + median line) for one split.
    Aggregates across all fc metrics by taking the mean per time series, so there is
    one figure per pred_eval metric only. Returns a list of (title, matplotlib_figure) tuples.
    """
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        HAS_SNS = True
    except Exception:
        import matplotlib.pyplot as plt  # type: ignore
        HAS_SNS = False

    splits = (dict_res or {}).get("splits", {}) or {}
    if split_name not in splits:
        return []
    pred_eval = (splits.get(split_name) or {}).get("pred_eval") or {}
    if not pred_eval:
        return []

    # Collect predictors and pred_metrics
    from afmo.core.registry import PREDICTORS
    predictors = list(PREDICTORS.keys())
    predictors.remove('best_method')
    #predictors = list(pred_eval.keys())
    pred_metrics = list({pm for p in predictors for pm in (pred_eval.get(p) or {}).keys()})

    figures = []
    for pm in pred_metrics:
        rows = []
        order = []
        for pred in predictors:
            # fc_to_ts: mapping fc_metric -> {ts_id -> value}
            fc_to_ts = ((pred_eval.get(pred) or {}).get(pm)) or {}
            if not isinstance(fc_to_ts, dict) or not fc_to_ts:
                continue

            # Determine the union of all time series keys across fc metrics
            ts_keys = set()
            for _, ts_map in (fc_to_ts or {}).items():
                if isinstance(ts_map, dict):
                    ts_keys.update(ts_map.keys())

            # For each time series, compute the mean across all fc metrics (finite values only)
            vals = []
            for ts in ts_keys:
                collected = []
                for _, ts_map in (fc_to_ts or {}).items():
                    v = (ts_map or {}).get(ts, None) if isinstance(ts_map, dict) else None
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(fv):
                        collected.append(fv)
                if collected:
                    vals.append(float(np.mean(collected)))

            if vals:
                order.append(pred)
                for v in vals:
                    rows.append({"predictor": pred, "value": v})

        if not rows or not order:
            continue

        df_long = pd.DataFrame(rows)

        # Build the figure (one figure per pred_metric)
        fig = plt.figure(figsize=(min(10, 4 + 0.4 * len(order)), 5), facecolor="white")
        ax = fig.gca(); ax.set_facecolor("white")

        if HAS_SNS:
            import seaborn as sns  # type: ignore
            sns.violinplot(
                data=df_long,
                x="predictor", y="value",
                # order=order,
                inner=None,
                cut=0,
                scale="width",
                width=0.8,
                linewidth=0.8,
                ax=ax,
            )
        else:
            # Fallback: use matplotlib violinplot
            parts = ax.violinplot([df_long[df_long["predictor"] == p]["value"].values for p in order],
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#2196f3")
                pc.set_edgecolor("#0d47a1")
                pc.set_alpha(0.55)

        # Median lines per predictor
        med = df_long.groupby("predictor")["value"].median()
        for i, pred in enumerate(order):
            y = med.get(pred, float("nan"))
            if not (y == y):  # NaN check
                continue
            ax.plot([i - 0.22, i + 0.22], [y, y], color="black", linewidth=2.2, solid_capstyle="butt")

        ax.set_title(f"{pm} ({dict_res['config']['model_name']})")
        ax.set_xlabel("Predictor")
        ax.set_ylabel("Value")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax.grid(False)
        try:
            import seaborn as sns  # type: ignore
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        except Exception:
            pass

        figures.append((f"{pm}", fig))

    return figures


def make_pred_eval_violins_for_split(dict_res: dict, split_name: str):
    """
    Build Seaborn/Matplotlib violin figures (only body + median line) for one split.
    Returns a list of (title, matplotlib_figure) tuples.
    """
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        HAS_SNS = True
    except Exception:
        import matplotlib.pyplot as plt  # type: ignore
        HAS_SNS = False

    splits = (dict_res or {}).get("splits", {}) or {}
    if split_name not in splits:
        return []
    pred_eval = (splits.get(split_name) or {}).get("pred_eval") or {}
    if not pred_eval:
        return []

    # Collect predictors, pred_metrics, and fc_metrics
    from afmo.core.registry import PREDICTORS
    predictors = list(PREDICTORS.keys())
    predictors.remove('best_method')
    # predictors = sorted(pred_eval.keys())
    pred_metrics = sorted({pm for p in predictors for pm in (pred_eval.get(p) or {}).keys()})
    fc_metrics = sorted({fc for p in predictors for pm, d1 in ((pred_eval.get(p) or {}).items()) for fc in (d1 or {}).keys()})

    figures = []
    for pm in pred_metrics:
        for fc in fc_metrics:
            rows = []
            order = []
            for pred in predictors:
                ts_map = ((pred_eval.get(pred) or {}).get(pm) or {}).get(fc) or {}
                vals = []
                for _, v in (ts_map or {}).items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(fv):
                        vals.append(fv)
                if vals:
                    order.append(pred)
                    for v in vals:
                        rows.append({"predictor": pred, "value": v})
            if not rows or not order:
                continue
            df_long = pd.DataFrame(rows)
            # Build the figure
            fig = plt.figure(figsize=(min(10, 4 + 0.4*len(order)), 5), facecolor="white") # 2 instead of 4
            ax = fig.gca(); ax.set_facecolor("white")
            if HAS_SNS:
                import seaborn as sns
                sns.violinplot(
                    data=df_long,
                    x="predictor", y="value",
                    inner=None,
                    cut=0,
                    scale="width",
                    width=0.8,
                    linewidth=0.8,
                    ax=ax,
                )
            else:
                # Fallback: use matplotlib violinplot
                parts = ax.violinplot([df_long[df_long["predictor"]==p]["value"].values for p in order],
                                      showmedians=False, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#2196f3")
                    pc.set_edgecolor("#0d47a1")
                    pc.set_alpha(0.55)
            # Median lines
            med = df_long.groupby("predictor")["value"].median()
            for i, pred in enumerate(order):
                y = med.get(pred, float("nan"))
                if not (y == y):  # NaN
                    continue
                ax.plot([i-0.22, i+0.22], [y, y], color="black", linewidth=2.2, solid_capstyle="butt")
            ax.set_title(f"{pm} × {fc} ({dict_res['config']['model_name']})")
            ax.set_xlabel("Predictor")
            ax.set_ylabel("Value")
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
            ax.grid(False)
            try:
                import seaborn as sns  # type: ignore
                sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
            except Exception:
                pass
            figures.append((f"{pm} × {fc}", fig))
    return figures


def _collect_pred_long(out: dict, split: Split) -> pd.DataFrame:
    """
    Build a long-form DataFrame with columns:
      predictor, fc_metric, stat, value
    where stat ∈ {"mean","lower","upper"} and each row is one time series value.
    """
    rows = []
    block = ((out or {}).get("splits") or {}).get(split, {}) or {}
    pred = block.get("pred") or {}
    if not isinstance(pred, dict) or not pred:
        return pd.DataFrame(columns=["predictor","fc_metric","stat","value"])

    from afmo.core.registry import PREDICTORS
    predictors = list(PREDICTORS.keys())
    predictors.remove('best_method')

    for predictor in predictors:
        ts_map = pred[predictor]
    #for predictor, ts_map in pred.items():
        if not isinstance(ts_map, dict):
            continue
        for ts_key, obj in ts_map.items():
            if not isinstance(obj, dict):
                continue
            for fc_metric, df in obj.items():
                try:
                    if isinstance(df, pd.DataFrame):
                        for stat in ("mean","lower","upper"):
                            if stat in df.columns:
                                v = df[stat].iloc[0]
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                                    rows.append({
                                        "predictor": str(predictor),
                                        "fc_metric": str(fc_metric),
                                        "stat": str(stat),
                                        "value": float(v),
                                    })
                    elif isinstance(df, dict):
                        # extremely defensive fallback if some predictors return dict-like
                        for stat in ("mean","lower","upper"):
                            v = df.get(stat, None)
                            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                                rows.append({
                                    "predictor": str(predictor),
                                    "fc_metric": str(fc_metric),
                                    "stat": str(stat),
                                    "value": float(v),
                                })
                except Exception:
                    # robust to any weird payload for one series/metric
                    continue
    if not rows:
        return pd.DataFrame(columns=["predictor","fc_metric","stat","value"])
    df_long = pd.DataFrame(rows)
    # ensure categorical ordering: predictors alphabetically, stat in fixed order
    preds = sorted(df_long["predictor"].unique(), key=str)
    df_long["predictor"] = pd.Categorical(df_long["predictor"], categories=preds, ordered=False)
    df_long["stat"] = pd.Categorical(df_long["stat"], categories=["mean","lower","upper"], ordered=False)
    return df_long

def violin_pred(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Grouped violin plots of predicted FC scores by predictor.
    One figure per fc_metric; within each predictor draw 3 violins: mean, lower, upper.
    """
    arts: list[PlotArtifact] = []
    try:
        import matplotlib.pyplot as plt
        HAS_SNS = False

        df_long = _collect_pred_long(out, split)
        if df_long.empty:
            return []

        fc_metrics = [m for m in df_long["fc_metric"].unique().tolist() if isinstance(m, str)]
        for fc in fc_metrics:
            df_fc = df_long[df_long["fc_metric"] == fc]
            if df_fc.empty:
                continue

            # reasonable width scaling based on number of predictors
            n_pred = int(df_fc["predictor"].nunique())
            width = min(12, 4 + 0.6 * n_pred)

            fig = plt.figure(figsize=(width, 5), facecolor="white")
            ax = fig.gca()
            ax.set_facecolor("white")

            if HAS_SNS:
                import seaborn as sns
                sns.violinplot(
                    data=df_fc,
                    x="predictor",
                    y="value",
                    hue="stat",
                    hue_order=["mean","lower","upper"],
                    dodge=True,
                    cut=0,
                    scale="width",
                    inner=None,
                    linewidth=0.8,
                    ax=ax,
                )
                # Place legend outside to save space
                ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
            else:
                # Fallback: draw three adjacent violins per predictor with matplotlib (grouped positions + legend)
                # Build data per predictor and stat, then place at grouped positions with small offsets
                from afmo.core.registry import PREDICTORS
                predictors = list(PREDICTORS.keys())
                predictors.remove('best_method')
                preds = predictors
                # preds = list(df_fc["predictor"].cat.categories if hasattr(df_fc["predictor"], "cat")
                #             else sorted(df_fc["predictor"].unique(), key=str))
                stat_order = ["mean", "lower", "upper"]

                # Group centers: one center per predictor
                centers = np.arange(len(preds), dtype=float)

                # Horizontal spread occupied by the whole group (tune if needed)
                group_spread = 0.7  # total width used by the 3 violins within a group
                # Offsets for each stat inside the group (symmetric around center)
                offsets = np.linspace(-group_spread/2, group_spread/2, len(stat_order))
                # Individual violin width
                violin_width = group_spread / (len(stat_order) + 0.5)

                # Color palette for stats (reuses Matplotlib default cycle)
                try:
                    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
                except Exception:
                    cycle = []
                if len(cycle) < len(stat_order):
                    cycle = cycle + ["C0", "C1", "C2"]  # simple fallback
                stat_colors = {s: cycle[i % len(cycle)] for i, s in enumerate(stat_order)}

                def _safe_vals(vals: np.ndarray) -> np.ndarray:
                    """Ensure violinplot gets at least two finite numbers for KDE stability."""
                    vals = np.asarray(vals, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        # give two NaNs; violinplot will simply skip drawing for that position
                        return np.array([np.nan, np.nan])
                    if vals.size == 1:
                        # duplicate singletons so KDE doesn't crash or look degenerate
                        return np.repeat(vals, 2)
                    return vals

                # Draw violins: one call per stat across all predictors, positioned with offsets
                for j, s in enumerate(stat_order):
                    data_s = []
                    for p in preds:
                        arr = df_fc[(df_fc["predictor"] == p) & (df_fc["stat"] == s)]["value"].values
                        data_s.append(_safe_vals(arr))
                    parts = ax.violinplot(
                        data_s,
                        positions=centers + offsets[j],
                        widths=violin_width,
                        showmedians=True,
                        showextrema=True
                    )
                    for body in parts.get("bodies", []):
                        body.set_facecolor(stat_colors[s])
                        body.set_edgecolor("black")
                        body.set_alpha(0.7)

                # One tick per predictor (group center), not per single violin
                ax.set_xticks(centers)
                ax.set_xticklabels([str(p) for p in preds], rotation=45, ha="right")

                # Build a legend for the three stats
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=stat_colors[s], alpha=0.7, label=s) for s in stat_order]
                ax.legend(handles=handles, title="", loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

                # Optional: slightly tighten horizontal margins
                ax.margins(x=0.05)

                # ensure y ticks are populated by a normal locator (in case something upstream changed it)
                import matplotlib.ticker as mticker
                ax.yaxis.set_major_locator(mticker.AutoLocator())
                ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

            ax.set_title(f"{fc} ({out['config']['model_name']})")
            ax.set_xlabel("Predictor")
            ax.set_ylabel("Predicted score")

            arts.append(PlotArtifact(
                id=f"violin_pred_{fc}".replace(" ", "_"),
                title=f"{fc}",
                kind="violin/pred",
                split=split,
                fig=fig,
                lib="mpl",
            ))
    except Exception:
        pass
    return arts

def pred_eval_violins(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Wrap the existing violin figures and fix axis labels to show the prediction metric on Y.
    """
    arts: list[PlotArtifact] = []
    try:
        for title, fig in make_pred_eval_violins_for_split_agg(out, split):
            # Axis relabel: title pattern "<pred_metric> × <fc_metric>"
            try:
                pm = str(title).split(" × ")[0] if " × " in str(title) else str(title)
                ax = fig.axes[0] if getattr(fig, "axes", None) else None
                if ax is not None:
                    ax.set_xlabel("Predictor")
                    ax.set_ylabel(pm)
            except Exception:
                pass
            arts.append(PlotArtifact(
                id=f"viol_{title}".replace(" ", "_"),
                title=title, kind="violin/pred_eval", split=split, fig=fig, lib="mpl"
            ))
    except Exception:
        pass
    try:
        for title, fig in make_pred_eval_violins_for_split(out, split):
            # Axis relabel: title pattern "<pred_metric> × <fc_metric>"
            try:
                pm = str(title).split(" × ")[0] if " × " in str(title) else str(title)
                ax = fig.axes[0] if getattr(fig, "axes", None) else None
                if ax is not None:
                    ax.set_xlabel("Predictor")
                    ax.set_ylabel(pm)
            except Exception:
                pass
            arts.append(PlotArtifact(
                id=f"viol_{title}".replace(" ", "_"),
                title=title, kind="violin/pred_eval", split=split, fig=fig, lib="mpl"
            ))
    except Exception:
        pass
    return arts