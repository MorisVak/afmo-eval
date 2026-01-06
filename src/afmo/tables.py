
"""
AFMo tables — core functions that produce pandas DataFrames for the GUI.
These functions take study result dictionaries (dict_res) and split names,
and return ready-to-render DataFrames for Spearman/Partial-Spearman associations
and Mann–Whitney comparisons between predictors.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd

def _flatten_fc_eval_features(block: dict) -> pd.DataFrame:
    """
    block = associations["spearman" or "partial_spearman"]["fc_eval"]
    Returns DataFrame with columns: feature, rho, pvalue, n, ci95_low, ci95_high, fc_metric
    """
    rows: List[Dict[str, Any]] = []
    for fc_metric, d in (block or {}).items():
        feats = (((d or {}).get("features")) or {})
        for feat, res in feats.items():
            ci = res.get("ci95") or [None, None]
            pval = res.get("pvalue", res.get("p"))
            rows.append({
                "feature": feat,
                "rho": res.get("rho"),
                "pvalue": pval,
                "n": res.get("n"),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "fc_metric": fc_metric,
            })
    return pd.DataFrame(rows)

def _flatten_pred_vs_features(block: dict) -> pd.DataFrame:
    """
    block = associations["spearman" or "partial_spearman"]["pred_vs_features"]
    Returns DataFrame with columns: predictor, feature, pred_metric, fc_metric, rho, pvalue, n, ci95_low, ci95_high
    """
    rows: List[Dict[str, Any]] = []
    for predictor, d in (block or {}).items():
        feats = (d or {}).get("features") or {}
        for feat, inner in feats.items():
            for pred_metric, m in (inner or {}).items():
                for fc_metric, res in (m or {}).items():
                    ci = res.get("ci95") or [None, None]
                    pval = res.get("pvalue", res.get("p"))
                    rows.append({
                        "predictor": predictor,
                        "feature": feat,
                        "pred_metric": pred_metric,
                        "fc_metric": fc_metric,
                        "rho": res.get("rho"),
                        "pvalue": pval,
                        "n": res.get("n"),
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                    })
    return pd.DataFrame(rows)

def _flatten_pred_vs_fc(block: dict) -> pd.DataFrame:
    """
    block = associations["spearman" or "partial_spearman"]["pred_vs_fc"]
    Returns DataFrame with columns: predictor, pred_metric, fc_metric, rho, pvalue, n, ci95_low, ci95_high
    """
    rows: List[Dict[str, Any]] = []
    for predictor, inner in (block or {}).items():
        for pred_metric, m in (inner or {}).items():
            for fc_metric, res in (m or {}).items():
                ci = res.get("ci95") or [None, None]
                pval = res.get("pvalue", res.get("p"))
                rows.append({
                    "predictor": predictor,
                    "pred_metric": pred_metric,
                    "fc_metric": fc_metric,
                    "rho": res.get("rho"),
                    "pvalue": pval,
                    "n": res.get("n"),
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                })
    return pd.DataFrame(rows)

def association_tables_for_split(dict_res: dict, split_name: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Produce six association tables for a given split:
      - Spearman:       fc_eval→features, pred_vs_features, pred_vs_fc
      - Partial Spearman: same three blocks
    Returns a dict like:
      {
        "spearman": {
            "fc_eval_features": df,
            "pred_vs_features": df,
            "pred_vs_fc": df
        },
        "partial_spearman": { ... }
      }
    """
    split = (dict_res or {}).get("splits", {}).get(split_name) or {}
    assoc = (split or {}).get("associations") or {}
    sp = assoc.get("spearman") or {}
    ps = assoc.get("partial_spearman") or {}

    out = {
        "spearman": {
            "fc_eval_features": _flatten_fc_eval_features(sp.get("fc_eval")),
            "pred_vs_features": _flatten_pred_vs_features(sp.get("pred_vs_features")),
            "pred_vs_fc": _flatten_pred_vs_fc(sp.get("pred_vs_fc")),
        },
        "partial_spearman": {
            "fc_eval_features": _flatten_fc_eval_features(ps.get("fc_eval")),
            "pred_vs_features": _flatten_pred_vs_features(ps.get("pred_vs_features")),
            "pred_vs_fc": _flatten_pred_vs_fc(ps.get("pred_vs_fc")),
        }
    }
    return out

def mannwhitney_tables_for_split(dict_res: dict, split_name: str) -> Dict[str, pd.DataFrame]:
    """
    Build predictor comparison tables via Mann–Whitney based on pred_eval.
    Returns a dict keyed by (pred_metric, fc_metric) concatenated as 'pred_metric|fc_metric' with a DataFrame:
      columns: predictor1, predictor2, pred_metric, fc_metric, pvalue, U, z, n1, n2,
               median1, iqr1, median2, iqr2, effect_r, better_at_5pct
    """
    from afmo.study import compute_pred_eval_stat
    split = (dict_res or {}).get("splits", {}).get(split_name) or {}
    pred_eval = split.get("pred_eval") or {}
    stats = compute_pred_eval_stat(pred_eval=pred_eval) if pred_eval else {}

    tables: Dict[str, pd.DataFrame] = {}
    # Flatten into rows
    rows_by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    if isinstance(stats, dict):
        for p1, d1 in stats.items():
            if not isinstance(d1, dict): continue
            for p2, d2 in d1.items():
                if p1 == p2 or not isinstance(d2, dict): continue
                for pm_name, inner in d2.items():
                    if not isinstance(inner, dict): continue
                    for fc_metric, res in inner.items():
                        key = (pm_name, fc_metric)
                        rows_by_key.setdefault(key, []).append({
                            "predictor1": p1,
                            "predictor2": p2,
                            "pred_metric": pm_name,
                            "fc_metric": fc_metric,
                            "pvalue": res.get("pvalue"),
                            "U": res.get("u_stat"),
                            "z": res.get("z_value"),
                            "n1": res.get("descriptives", {}).get("group1", {}).get("n"),
                            "n2": res.get("descriptives", {}).get("group2", {}).get("n"),
                            "median1": res.get("descriptives", {}).get("group1", {}).get("median"),
                            "iqr1": res.get("descriptives", {}).get("group1", {}).get("iqr"),
                            "median2": res.get("descriptives", {}).get("group2", {}).get("median"),
                            "iqr2": res.get("descriptives", {}).get("group2", {}).get("iqr"),
                            "effect_r": res.get("effect_r"),
                            "better_at_5pct": res.get("better_at_5pct"),
                        })
    for key, rows in rows_by_key.items():
        pm, fc = key
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["pvalue"], ascending=True, na_position="last")
        tables[f"{pm}|{fc}"] = df
    return tables
