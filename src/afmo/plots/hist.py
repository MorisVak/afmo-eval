from __future__ import annotations
from typing import Iterable, Tuple, Any, Dict
from .base import PlotArtifact, Split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# central place to keep consistent sizes and styling
_MPL_SIZES = dict(title=15, label=12, tick=11)

def _style_axes(ax, *, title: str, xlabel: str, ylabel: str, grid: bool = False):
    ax.set_title(title, fontsize=_MPL_SIZES["title"])
    ax.set_xlabel(xlabel, fontsize=_MPL_SIZES["label"])
    ax.set_ylabel(ylabel, fontsize=_MPL_SIZES["label"])
    ax.tick_params(labelsize=_MPL_SIZES["tick"])
    ax.set_facecolor("white")
    ax.grid(grid, alpha=0.25)
    if ax.figure is not None:
        ax.figure.set_facecolor("white")

def _flatten_features(out: dict, split: str) -> pd.DataFrame:
    """Return long-form DataFrame with columns [ts, feature, value]."""
    df = out["splits"][split]["X_sel"]
    df_long = df.copy()
    df_long = df_long.assign(ts=df_long.index)
    df_long = df_long.melt(id_vars="ts", var_name="feature", value_name="value")
    return df_long

def _flatten_fc_eval(out: dict, split: str) -> pd.DataFrame:
    """Return long-form DataFrame with columns [ts, metric, value]."""
    fc_eval = out["splits"][split]["fc_eval"]
    rows = []
    for metric, per_ts in fc_eval.items():
        for ts, val in per_ts.items():
            rows.append({"ts": ts, "metric": metric, "value": val})
    return pd.DataFrame(rows)

def _flatten_pred(out: dict, split: str) -> pd.DataFrame:
    """
    Return long-form DataFrame with columns [ts, predictor, metric, value].
    Uses the 'mean' from each prediction band.
    """
    pred = out["splits"][split]["pred"]
    rows = []
    for predictor, per_ts in pred.items():
        for ts, per_metric in per_ts.items():
            for metric, band in per_metric.items():
                if isinstance(band, dict) and "mean" in band:
                    rows.append({"ts": ts, "predictor": predictor, "metric": metric, "value": band["mean"]})
    return pd.DataFrame(rows)

def _flatten_pred_eval(out: dict, split: str) -> pd.DataFrame:
    """Return long-form DataFrame with columns [ts, predictor, pred_metric, metric, value]."""
    pred_eval = out["splits"][split]["pred_eval"]
    rows = []
    for predictor, per_pred_metric in pred_eval.items():
        for pred_metric, per_metric in per_pred_metric.items():
            for metric, per_ts in per_metric.items():
                for ts, val in per_ts.items():
                    rows.append({
                        "ts": ts,
                        "predictor": predictor,
                        "pred_metric": pred_metric,
                        "metric": metric,
                        "value": val
                    })
    return pd.DataFrame(rows)

def _flatten_best_method(out: dict, split: str) -> pd.DataFrame:
    """Return DataFrame with columns [ts, best_method]."""
    bm = out["splits"][split]["best_method"]
    return pd.DataFrame([{"ts": ts, "best_method": method} for ts, method in bm.items()])

def _flatten_predicted_best_method(out: dict, split: str) -> pd.DataFrame:
    """Return DataFrame with columns [ts, best_method_pred] from pred_ranks (lower is better).
    Tie-breaker: pick lexicographically first predictor among ties.
    """
    rd = out["splits"][split]["rank_deviation"]
    pr = (rd or {}).get("per_ts", {}).get("pred_ranks", {})
    rows = []
    for ts, ranks in (pr or {}).items():
        if not isinstance(ranks, dict) or not ranks:
            continue
        vals = [(p, float(v)) for p, v in ranks.items() if v is not None and np.isfinite(v)]
        if not vals:
            continue
        min_val = min(v for _, v in vals)
        tied = sorted([p for p, v in vals if abs(v - min_val) < 1e-12], key=str)
        rows.append({"ts": ts, "best_method_pred": tied[0]})
    return pd.DataFrame(rows)

def _all_predictors_available(out: dict, split: str) -> list[str]:
    """Collect all predictor names present in this split, robust across multiple sources."""
    preds = set()
    # primary: predictors actually run
    try:
        preds |= set((out["splits"][split]["pred"] or {}).keys())
    except Exception:
        pass
    # any seen in real ranks
    try:
        rr = ((out["splits"][split].get("rank_deviation") or {}).get("per_ts") or {}).get("real_ranks", {})
        for _ts, d in (rr or {}).items():
            if isinstance(d, dict):
                preds |= set(d.keys())
    except Exception:
        pass
    # any that appeared as best_method
    try:
        bm = (out["splits"][split].get("best_method") or {})
        preds |= set(bm.values())
    except Exception:
        pass
    return sorted(preds, key=str)

def make_plotly_figures_nested(out: dict, splits: Tuple[str, ...] = ("train", "val", "test")) -> Dict[str, Dict[str, Any]]:
    """
    Returns the same nested dict structure as before, but all figures are Matplotlib figures:
      figs[split]["features"][<feature>] = histogram figure
      figs[split]["fc_eval"][<metric>] = histogram figure
      figs[split]["pred"][<metric>] = histogram (mean values), colored by predictor
      figs[split]["pred_eval"][<pred_metric>][<metric>] = histogram, colored by predictor
      figs[split]["best_method"]["distribution"] = bar chart of frequencies
      figs["test"]["best_method_eval"]["quadrant"] = quadrant scatter (if available)
    """
    figs: Dict[str, Dict[str, Any]] = {}

    from afmo.core.registry import PREDICTORS
    predictors = list(PREDICTORS.keys())
    predictors.remove('best_method')

    for split in splits:
        figs[split] = {}

        # features: one histogram per feature
        try:
            df_feat = _flatten_features(out, split)
            if not df_feat.empty:
                figs[split]["features"] = {}
                for feature, d in df_feat.groupby("feature"):
                    d = d.rename(columns={"value": feature}, errors="raise")
                    vals = d[feature].dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
                    ax.hist(vals, bins=40, alpha=0.9)
                    _style_axes(
                        ax,
                        title=f"Feature histogram: {feature}",
                        xlabel=str(feature),
                        ylabel="count",
                    )
                    fig.tight_layout()
                    figs[split]["features"][str(feature)] = fig
        except KeyError:
            pass

        # fc_eval: one histogram per metric
        try:
            df_fce = _flatten_fc_eval(out, split)
            if not df_fce.empty:
                figs[split]["fc_eval"] = {}
                for metric, d in df_fce.groupby("metric"):
                    d = d.rename(columns={"value": metric}, errors="raise")
                    vals = d[metric].dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
                    ax.hist(vals, bins=40, alpha=0.9)
                    _style_axes(
                        ax,
                        title=f"Forecast accuracy histogram: {metric} ({out['config']['model_name']})",
                        xlabel=str(metric),
                        ylabel="count",
                    )
                    fig.tight_layout()
                    figs[split]["fc_eval"][str(metric)] = fig
        except KeyError:
            pass

        try:
            df_bm = _flatten_best_method(out, split)
            #all_preds = _all_predictors_available(out, split)
            all_preds = predictors

            # start with zeros for all predictors, then fill actual counts
            base_counts = pd.Series(0, index=pd.Index(all_preds, dtype=object))
            if not df_bm.empty:
                vc = df_bm["best_method"].value_counts()
                base_counts = base_counts.add(vc, fill_value=0).astype(int)
                base_counts = base_counts.reindex(all_preds)

            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
            xs = np.arange(len(base_counts))
            ys = base_counts.to_numpy()
            ax.bar(xs, ys)
            ax.set_xticks(xs, base_counts.index.astype(str).tolist(), rotation=30, ha="right")
            for x, y in zip(xs, ys):
                ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])
            _style_axes(
                ax,
                title=f"Best estimator histogram (actual)  ({out['config']['model_name']})",
                xlabel="predictor",
                ylabel="count",
            )
            fig.tight_layout()
            figs.setdefault(split, {})
            figs[split].setdefault("best_method", {})
            figs[split]["best_method"]["distribution_actual"] = fig
        except KeyError:
            pass

        try:
            df_bm = _flatten_predicted_best_method(out, split)
            #all_preds = _all_predictors_available(out, split)
            all_preds = predictors
            # zeros for all predictors, then fill actual predicted counts
            base_counts = pd.Series(0, index=pd.Index(all_preds, dtype=object))
            if df_bm is not None and not df_bm.empty:
                vc = df_bm["best_method_pred"].value_counts()
                base_counts = base_counts.add(vc, fill_value=0).astype(int)
                base_counts = base_counts.reindex(all_preds)

            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
            xs = np.arange(len(base_counts))
            ys = base_counts.to_numpy()
            ax.bar(xs, ys)
            ax.set_xticks(xs, base_counts.index.astype(str).tolist(), rotation=30, ha="right")
            for x, y in zip(xs, ys):
                ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])
            _style_axes(
                ax,
                title=f"Best estimator histogram (predicted) ({out['config']['model_name']})",
                xlabel="predictor (predicted best)",
                ylabel="count",
            )
            fig.tight_layout()
            figs.setdefault(split, {})
            figs[split].setdefault("best_method", {})
            figs[split]["best_method"]["distribution_pred"] = fig
        except KeyError:
            pass

        # try:
        #     rd = out["splits"][split]["rank_deviation"]
        #     hist = rd.get("hist_int", {})
        #     if isinstance(hist, dict) and len(hist) > 0:
        #         # ensure integer-sorted bins
        #         rows = []
        #         for k, v in hist.items():
        #             try:
        #                 ki = int(k)
        #                 vi = int(v)
        #             except Exception:
        #                 continue
        #             rows.append((ki, vi))
        #         if rows:
        #             rows.sort(key=lambda t: t[0])
        #             devs = [t[0] for t in rows]
        #             counts = [t[1] for t in rows]

        #             fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        #             xs = np.arange(len(devs))
        #             ax.bar(xs, counts)
        #             ax.set_xticks(xs, [str(d) for d in devs], rotation=0, ha="center")
        #             for x, y in zip(xs, counts):
        #                 ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])
        #             _style_axes(
        #                 ax,
        #                 title=f"Rank deviation (predicted − actual) ({out['config']['model_name']})",
        #                 xlabel="rank deviation",
        #                 ylabel="count",
        #             )
        #             fig.tight_layout()
        #             figs.setdefault(split, {})
        #             figs[split].setdefault("rank_deviation", {})
        #             figs[split]["best_method"]["rank_deviation"] = fig
        # except KeyError:
        #     pass

        # try:
        #     rd = out["splits"][split]["rank_deviation"]
        #     dev_map = ((rd or {}).get("per_ts") or {}).get("deviations", {})
        #     meta = (rd or {}).get("meta", {})
        #     bin_min = int(meta.get("bin_min", -10))
        #     bin_max = int(meta.get("bin_max", 10))

        #     # collect predictors present across all series
        #     predictors = set()
        #     for _ts, inner in (dev_map or {}).items():
        #         if isinstance(inner, dict):
        #             predictors |= set(inner.keys())

        #     #figs.setdefault(split, {})
        #     figs[split].setdefault("best_method_eval", {})
        #     figs[split]["best_method_eval"].setdefault("rank_deviation_all", {})

        #     bins_sorted = list(range(bin_min, bin_max + 1))

        #     for predictor in sorted(predictors, key=str):
        #         # build integer-binned counts for this predictor
        #         counts = {k: 0 for k in bins_sorted}
        #         for _ts, inner in dev_map.items():
        #             if not isinstance(inner, dict):
        #                 continue
        #             v = inner.get(predictor, None)
        #             if v is None:
        #                 continue
        #             try:
        #                 dev_int = int(np.rint(float(v)))
        #             except Exception:
        #                 continue
        #             # clamp to bin range just in case
        #             dev_int = max(bin_min, min(bin_max, dev_int))
        #             counts[dev_int] = counts.get(dev_int, 0) + 1

        #         ys = [int(counts.get(k, 0)) for k in bins_sorted]
        #         if sum(ys) == 0:
        #             # nothing to plot for this predictor
        #             continue

        #         fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
        #         xs = np.arange(len(bins_sorted))
        #         ax.bar(xs, ys)
        #         ax.set_xticks(xs, [str(k) for k in bins_sorted], rotation=0, ha="center")
        #         for x, y in zip(xs, ys):
        #             if y > 0:
        #                 ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])

        #         _style_axes(
        #             ax,
        #             title=f"Rank deviation (predicted − actual) for {predictor} ({out['config']['model_name']})",
        #             xlabel="rank deviation",
        #             ylabel="count",
        #         )
        #         fig.tight_layout()
        #         figs[split]["best_method_eval"]["rank_deviation_all"][predictor] = fig
        # except KeyError:
        #     pass

        # from afmo.core.registry import PREDICTORS
        # predictors = list(PREDICTORS.keys())
        # predictors.remove('best_method')

        # try:
        #     rr = out["splits"][split]["rank_deviation"]["per_ts"]["real_ranks"]
        #     meta = (out["splits"][split]["rank_deviation"] or {}).get("meta", {})
        #     P = int(meta.get("P", 0)) if isinstance(meta.get("P", 0), (int, float)) else 0

        #     # Collect predictors seen across series
        #     # predictors = set()
        #     # for _ts, ranks in (rr or {}).items():
        #     #     if isinstance(ranks, dict):
        #     #         predictors |= set(ranks.keys())
            

        #     #predictors = sorted(predictors, key=str)
        #     ys_by_pred = {p: [] for p in predictors}
        #     for _ts, ranks in rr.items():
        #         if not isinstance(ranks, dict):
        #             continue
        #         for p in predictors:
        #             v = ranks.get(p, None)
        #             if v is None:
        #                 continue
        #             try:
        #                 vf = float(v)
        #                 if np.isfinite(vf):
        #                     ys_by_pred[p].append(vf)
        #             except Exception:
        #                 continue

        #     data = [ys_by_pred[p] for p in predictors]

        #     fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        #     positions = np.arange(1, len(predictors) + 1)
        #     vp = ax.violinplot(
        #         data,
        #         positions=positions,
        #         showmeans=False,
        #         showmedians=True,
        #         showextrema=False,
        #     )

        #     # x-axis with predictor labels
        #     ax.set_xticks(positions, [str(p) for p in predictors], rotation=30, ha="right")

        #     # y-limits to rank range if P known
        #     if P and P > 0:
        #         ax.set_ylim(0.5, P + 0.5)

        #     _style_axes(
        #         ax,
        #         title=f"Ranks violinplot (actual) ({out['config']['model_name']})",
        #         xlabel="predictor",
        #         ylabel="real rank (1 = best)",
        #     )
        #     fig.tight_layout()

        #     figs[split]["best_method"]["violin_actual"] = fig
        # except KeyError:
        #     pass

        # try:
        #     rr = out["splits"][split]["rank_deviation"]["per_ts"]["pred_ranks"]
        #     meta = (out["splits"][split]["rank_deviation"] or {}).get("meta", {})
        #     P = int(meta.get("P", 0)) if isinstance(meta.get("P", 0), (int, float)) else 0

        #     # Collect predictors seen across series
        #     # predictors = set()
        #     # for _ts, ranks in (rr or {}).items():
        #     #     if isinstance(ranks, dict):
        #     #         predictors |= set(ranks.keys())

        #     #predictors = sorted(predictors, key=str)
        #     ys_by_pred = {p: [] for p in predictors}
        #     for _ts, ranks in rr.items():
        #         if not isinstance(ranks, dict):
        #             continue
        #         for p in predictors:
        #             v = ranks.get(p, None)
        #             if v is None:
        #                 continue
        #             try:
        #                 vf = float(v)
        #                 if np.isfinite(vf):
        #                     ys_by_pred[p].append(vf)
        #             except Exception:
        #                 continue

        #     data = [ys_by_pred[p] for p in predictors]

        #     fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        #     positions = np.arange(1, len(predictors) + 1)
        #     vp = ax.violinplot(
        #         data,
        #         positions=positions,
        #         showmeans=False,
        #         showmedians=True,
        #         showextrema=False,
        #     )

        #     # x-axis with predictor labels
        #     ax.set_xticks(positions, [str(p) for p in predictors], rotation=30, ha="right")

        #     # y-limits to rank range if P known
        #     if P and P > 0:
        #         ax.set_ylim(0.5, P + 0.5)

        #     _style_axes(
        #         ax,
        #         title=f"Ranks violinplot (predicted) ({out['config']['model_name']})",
        #         xlabel="predictor",
        #         ylabel="predicted rank (1 = best)",
        #     )
        #     fig.tight_layout()

        #     figs[split]["best_method"]["violin_predicted"] = fig
        # except KeyError:
        #     pass

        try:
            rd = out["splits"][split]["rank_deviation"]
            hist = rd.get("hist_int", {})
            if isinstance(hist, dict) and len(hist) > 0:
                # ensure integer-sorted bins
                rows = []
                for k, v in hist.items():
                    try:
                        ki = int(k)
                        vi = int(v)
                    except Exception:
                        continue
                    rows.append((ki, vi))
                if rows:
                    # limit rank deviation range based on number of predictors
                    max_dev = len(predictors) - 1  # e.g. 7 predictors -> -6..6
                    rows = [(k, v) for (k, v) in rows if -max_dev <= k <= max_dev]

                    rows.sort(key=lambda t: t[0])
                    devs = [t[0] for t in rows]
                    counts = [t[1] for t in rows]

                    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
                    xs = np.arange(len(devs))
                    ax.bar(xs, counts)
                    ax.set_xticks(xs, [str(d) for d in devs], rotation=0, ha="center")
                    for x, y in zip(xs, counts):
                        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])
                    _style_axes(
                        ax,
                        title=f"Rank deviation (predicted − actual) ({out['config']['model_name']})",
                        xlabel="rank deviation",
                        ylabel="count",
                    )
                    fig.tight_layout()
                    figs.setdefault(split, {})
                    figs[split].setdefault("rank_deviation", {})
                    figs[split]["best_method"]["rank_deviation"] = fig
        except KeyError:
            pass

        try:
            rd = out["splits"][split]["rank_deviation"]
            dev_map = ((rd or {}).get("per_ts") or {}).get("deviations", {})
            meta = (rd or {}).get("meta", {})
            bin_min = int(meta.get("bin_min", -10))
            bin_max = int(meta.get("bin_max", 10))

            # collect predictors present across all series
            predictors = set()
            for _ts, inner in (dev_map or {}).items():
                if isinstance(inner, dict):
                    predictors |= set(inner.keys())

            # limit bin range based on number of predictors
            if predictors:
                max_dev = len(predictors) - 1  # e.g. 7 predictors -> -6..6
                bin_min = max(bin_min, -max_dev)
                bin_max = min(bin_max, max_dev)

            #figs.setdefault(split, {})
            figs[split].setdefault("best_method_eval", {})
            figs[split]["best_method_eval"].setdefault("rank_deviation_all", {})

            bins_sorted = list(range(bin_min, bin_max + 1))

            for predictor in sorted(predictors, key=str):
                # build integer-binned counts for this predictor
                counts = {k: 0 for k in bins_sorted}
                for _ts, inner in dev_map.items():
                    if not isinstance(inner, dict):
                        continue
                    v = inner.get(predictor, None)
                    if v is None:
                        continue
                    try:
                        dev_int = int(np.rint(float(v)))
                    except Exception:
                        continue
                    # clamp to bin range just in case
                    dev_int = max(bin_min, min(bin_max, dev_int))
                    counts[dev_int] = counts.get(dev_int, 0) + 1

                ys = [int(counts.get(k, 0)) for k in bins_sorted]
                if sum(ys) == 0:
                    # nothing to plot for this predictor
                    continue

                fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
                xs = np.arange(len(bins_sorted))
                ax.bar(xs, ys)
                ax.set_xticks(xs, [str(k) for k in bins_sorted], rotation=0, ha="center")
                for x, y in zip(xs, ys):
                    if y > 0:
                        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=_MPL_SIZES["tick"])

                _style_axes(
                    ax,
                    title=f"Rank deviation (predicted − actual) for {predictor} ({out['config']['model_name']})",
                    xlabel="rank deviation",
                    ylabel="count",
                )
                fig.tight_layout()
                figs[split]["best_method_eval"]["rank_deviation_all"][predictor] = fig
        except KeyError:
            pass

        from afmo.core.registry import PREDICTORS
        predictors = list(PREDICTORS.keys())
        predictors.remove('best_method')

        try:
            rr = out["splits"][split]["rank_deviation"]["per_ts"]["real_ranks"]
            meta = (out["splits"][split]["rank_deviation"] or {}).get("meta", {})
            P = int(meta.get("P", 0)) if isinstance(meta.get("P", 0), (int, float)) else 0

            ys_by_pred = {p: [] for p in predictors}
            for _ts, ranks in rr.items():
                if not isinstance(ranks, dict):
                    continue
                for p in predictors:
                    v = ranks.get(p, None)
                    if v is None:
                        continue
                    try:
                        vf = float(v)
                        if np.isfinite(vf):
                            ys_by_pred[p].append(vf)
                    except Exception:
                        continue

            data = [ys_by_pred[p] for p in predictors]

            fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
            positions = np.arange(1, len(predictors) + 1)
            vp = ax.violinplot(
                data,
                positions=positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            # x-axis with predictor labels
            ax.set_xticks(positions, [str(p) for p in predictors], rotation=30, ha="right")

            # y-limits to rank range if P known
            if P and P > 0:
                ax.set_ylim(0.5, P + 0.5)

            _style_axes(
                ax,
                title=f"Ranks violinplot (actual) ({out['config']['model_name']})",
                xlabel="predictor",
                ylabel="real rank (1 = best)",
            )
            fig.tight_layout()

            figs[split]["best_method"]["violin_actual"] = fig
        except KeyError:
            pass

        try:
            rr = out["splits"][split]["rank_deviation"]["per_ts"]["pred_ranks"]
            meta = (out["splits"][split]["rank_deviation"] or {}).get("meta", {})
            P = int(meta.get("P", 0)) if isinstance(meta.get("P", 0), (int, float)) else 0

            ys_by_pred = {p: [] for p in predictors}
            for _ts, ranks in rr.items():
                if not isinstance(ranks, dict):
                    continue
                for p in predictors:
                    v = ranks.get(p, None)
                    if v is None:
                        continue
                    try:
                        vf = float(v)
                        if np.isfinite(vf):
                            ys_by_pred[p].append(vf)
                    except Exception:
                        continue

            data = [ys_by_pred[p] for p in predictors]

            fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
            positions = np.arange(1, len(predictors) + 1)
            vp = ax.violinplot(
                data,
                positions=positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            # x-axis with predictor labels
            ax.set_xticks(positions, [str(p) for p in predictors], rotation=30, ha="right")

            # y-limits to rank range if P known
            if P and P > 0:
                ax.set_ylim(0.5, P + 0.5)

            _style_axes(
                ax,
                title=f"Ranks violinplot (predicted) ({out['config']['model_name']})",
                xlabel="predictor",
                ylabel="predicted rank (1 = best)",
            )
            fig.tight_layout()

            figs[split]["best_method"]["violin_predicted"] = fig
        except KeyError:
            pass
        
    return figs

def _yield_from_nested(nested: dict, split: str, group: str, kind: str) -> Iterable[PlotArtifact]:
    block = (nested.get(split) or {}).get(group) or {}
    if not block:
        return []
    out = []
    if group in ("features", "fc_eval", "pred"):
        for key, fig in block.items():
            out.append(PlotArtifact(
                id=f"{group}_{key}",
                title=str(key),
                kind=kind,
                split=split, fig=fig, lib="mpl"
            ))
    elif group == "pred_eval":
        for pred_metric, sub in block.items():
            for metric, sub2 in (sub or {}).items():
                for predictor, fig in (sub2 or {}).items():
                    title = f"{pred_metric}__{metric}__{predictor}"
                    out.append(PlotArtifact(
                        id=f"pred_eval_{pred_metric}__{metric}__{predictor}",
                        title=title,
                        kind=kind,
                        split=split, fig=fig, lib="mpl"
                    ))
    elif group == "best_method":
        for k, fig in block.items():
            out.append(PlotArtifact(
                id=f"best_method_{k}",
                title=str(k),
                kind=kind,
                split=split, fig=fig, lib="mpl"
            ))
    elif group == "best_method_eval":
        for pred_metric, sub in block.items():
            for k, fig in sub.items():
                out.append(PlotArtifact(
                    id=f"best_method_eval_{k}",
                    title=str(k),
                    kind=kind,
                    split=split, fig=fig, lib="mpl"
                ))
    return out

def features_hist(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    arts = _yield_from_nested(nested, split, group="features", kind="features/hist")
    for a in arts:
        try:
            a.fig.update_layout(xaxis_title=a.title, yaxis_title="Count")
        except Exception:
            pass
    return arts

def fc_eval_hist(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    arts = _yield_from_nested(nested, split, group="fc_eval", kind="fc_eval/hist")
    for a in arts:
        try:
            a.fig.update_layout(xaxis_title=a.title, yaxis_title="Count")
        except Exception:
            pass
    return arts

def pred_hist(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    arts = _yield_from_nested(nested, split, group="pred", kind="pred/hist")
    for a in arts:
        try:
            a.fig.update_layout(xaxis_title=a.title, yaxis_title="Count")
        except Exception:
            pass
    return arts

def pred_eval_hist(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    arts = _yield_from_nested(nested, split, group="pred_eval", kind="pred_eval/hist")
    for a in arts:
        try:
            parts = str(a.title).split("__")
            xlab = parts[-1] if parts else a.title
            a.fig.update_layout(xaxis_title=xlab, yaxis_title="Count")
        except Exception:
            pass
    return arts

def best_method_dist(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    return _yield_from_nested(nested, split, group="best_method", kind="best_method/dist")

def best_method_quadrant(out: dict, split: Split):
    nested = make_plotly_figures_nested(out, splits=(split,))
    return _yield_from_nested(nested, split, group="best_method_eval", kind="best_method/quadrant")