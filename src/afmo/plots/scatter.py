from __future__ import annotations
from typing import Iterable, Tuple, Any, Dict
from .base import PlotArtifact, Split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter_features(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Create 3D scatter plots of the clustering features and cluster assignments.
    First figure shows only the 3D feature space.
    Second figure colors by df_cluster_train.
    Third figure colors by df_cluster_target.
    """

    arts: list[PlotArtifact] = []

    try:
        # feature matrix used for clustering (columns are 3 features, indices are time series labels)
        X_train = out["splits"][split]["X_train"]

        # convert to DataFrame-like access if needed
        # ensure we have 3 columns
        if hasattr(X_train, "iloc"):
            x = X_train.iloc[:, 0].values
            y = X_train.iloc[:, 1].values
            z = X_train.iloc[:, 2].values
            idx = X_train.index
        else:
            # assume numpy-like
            x = np.asarray(X_train)[:, 0]
            y = np.asarray(X_train)[:, 1]
            z = np.asarray(X_train)[:, 2]
            idx = None

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=20)

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title("Features of training data")

        arts.append(PlotArtifact(
            id=f"{split}_scatter_features".replace(" ", "_"),
            title=f"Features of training data",
            kind="scatter/features",
            split=split,
            fig=fig,
            lib="mpl",
        ))
    except Exception:
        pass

    return arts

def scatter_clustering(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    3D scatter plots of clustering with fixed, reproducible colors.
    -1 -> black
    0..50 -> categorical, clearly distinguishable
    51..200 -> continuous
    All colors are RGBA.
    Axes are fixed to [0, 1] on x, y, z.
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    arts: list[PlotArtifact] = []

    LABEL_MIN = -1
    LABEL_MAX = 200

    # build categorical colors for 0..50
    tab10 = list(plt.cm.get_cmap("tab10").colors)
    tab20 = list(plt.cm.get_cmap("tab20").colors)
    categorical = tab10 + tab20
    while len(categorical) < 51:
        categorical += tab20
    categorical = categorical[:51]

    # fixed mapping label -> RGBA
    label_to_color: dict[int, tuple] = {}
    label_to_color[-1] = mcolors.to_rgba((0.0, 0.0, 0.0))  # noise

    for lab in range(0, 51):
        label_to_color[lab] = mcolors.to_rgba(categorical[lab])

    cont_cmap = plt.cm.get_cmap("nipy_spectral")
    for lab in range(51, LABEL_MAX + 1):
        t = (lab - 51) / max(1, LABEL_MAX - 51)
        label_to_color[lab] = cont_cmap(t)

    fallback_color = mcolors.to_rgba((0.5, 0.5, 0.5))

    df_cluster_train_dict = out["splits"][split].get("df_cluster_train", {}) or {}
    df_cluster_target_dict = out["splits"][split].get("df_cluster_target", {}) or {}

    def get_xyz(out, split):
        if split == "train":
            X = out["splits"][split]["X_train"]
        else:
            X = out["splits"][split]["X_target"]

        if hasattr(X, "iloc"):
            x = X.iloc[:, 0].to_numpy()
            y = X.iloc[:, 1].to_numpy()
            z = X.iloc[:, 2].to_numpy()
            feature_index = X.index
        else:
            X = np.asarray(X)
            x = X[:, 0]
            y = X[:, 1]
            z = X[:, 2]
            feature_index = None
        return x, y, z, feature_index

    def label_to_rgba(lab):
        try:
            li = int(lab)
        except Exception:
            return fallback_color
        if li < LABEL_MIN or li > LABEL_MAX:
            return fallback_color
        return label_to_color.get(li, fallback_color)

    x, y, z, feature_index = get_xyz(out, split)

    # train
    for predictor, df_cluster_train in df_cluster_train_dict.items():
        try:
            if hasattr(df_cluster_train, "reindex") and feature_index is not None:
                labels = (
                    df_cluster_train["label"]
                    .reindex(feature_index)   # align to X
                    .fillna(-1)               # missing -> noise
                    .astype(int)              # make sure we have ints
                    .to_numpy()
                )
            else:
                labels = np.asarray(df_cluster_train["label"].to_numpy(), dtype=int)

            n = min(len(x), len(labels))
            x_ = x[:n]
            y_ = y[:n]
            z_ = z[:n]
            labels = labels[:n]

            colors = [label_to_rgba(lab) for lab in labels]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_, y_, z_, c=colors, s=20)

            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title(f"Cluster Assignment train for {predictor}")

            # fixed axis limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass

            arts.append(
                PlotArtifact(
                    id=f"scatter_train_{predictor}".replace(" ", "_"),
                    title=f"Cluster Assignment train for {predictor}",
                    kind="scatter/cluster",
                    split=split,
                    fig=fig,
                    lib="mpl",
                )
            )
        except Exception:
            pass

    # target
    for predictor, df_cluster_target in df_cluster_target_dict.items():
        try:
            if hasattr(df_cluster_target, "reindex") and feature_index is not None:
                labels = (
                    df_cluster_target["label"]
                    .reindex(feature_index)
                    .fillna(-1)
                    .astype(int)
                    .to_numpy()
                )
            else:
                labels = np.asarray(df_cluster_target["label"].to_numpy(), dtype=int)

            n = min(len(x), len(labels))
            x_ = x[:n]
            y_ = y[:n]
            z_ = z[:n]
            labels = labels[:n]

            colors = [label_to_rgba(lab) for lab in labels]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_, y_, z_, c=colors, s=20)

            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title(f"Cluster Assignment target for {predictor}")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass

            arts.append(
                PlotArtifact(
                    id=f"scatter_target_{predictor}".replace(" ", "_"),
                    title=f"Cluster Assignment target for {predictor}",
                    kind="scatter/cluster",
                    split=split,
                    fig=fig,
                    lib="mpl",
                )
            )
        except Exception:
            pass

    return arts


def hist_clustering(out: dict, split: Split) -> Iterable[PlotArtifact]:

    arts: list[PlotArtifact] = []

    # train histograms (only if we are actually on train split)
    try:
        if split != "train":
            raise Exception()
        df_cluster_train_dict = out["splits"][split]["df_cluster_train"]
        for predictor, df_cluster_train in df_cluster_train_dict.items():
            labels = df_cluster_train["label"]
            counts = labels.value_counts().sort_index()

            fig, ax = plt.subplots()
            ax.bar(range(len(counts)), counts.values)

            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x) for x in counts.index], rotation=45, ha="right")

            ax.set_xlabel("Cluster label")
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram cluster assignment train for {predictor}")

            arts.append(PlotArtifact(
                id=f"{split}_hist_cluster_train_{predictor}".replace(" ", "_"),
                title=f"Histogram cluster assignment train for {predictor}",
                kind="hist/cluster",
                split=split,
                fig=fig,
                lib="mpl",
            ))
    except Exception:
        pass

    # target histograms, but make sure we show all train labels
    try:
        df_cluster_target_dict = out["splits"][split]["df_cluster_target"]

        # try to get the train dict for label reference
        train_cluster_by_pred = out["splits"].get("train", {}).get("df_cluster_train", {})

        for predictor, df_cluster_target in df_cluster_target_dict.items():
            target_labels = df_cluster_target["label"]
            target_counts = target_labels.value_counts().sort_index()

            # get train labels for this predictor (if available)
            if predictor in train_cluster_by_pred:
                train_labels = train_cluster_by_pred[predictor]["label"]
                train_label_index = train_labels.value_counts().sort_index().index
                # reindex target counts to include all train labels
                target_counts = target_counts.reindex(train_label_index, fill_value=0)
            # else: keep target_counts as is

            fig, ax = plt.subplots()
            ax.bar(range(len(target_counts)), target_counts.values)

            ax.set_xticks(range(len(target_counts)))
            ax.set_xticklabels([str(x) for x in target_counts.index], rotation=45, ha="right")

            ax.set_xlabel("Cluster label")
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram cluster assignment target for {predictor}")

            arts.append(PlotArtifact(
                id=f"{split}_hist_cluster_target_{predictor}".replace(" ", "_"),
                title=f"Histogram Cluster Assignment target for {predictor}",
                kind="hist/cluster",
                split=split,
                fig=fig,
                lib="mpl",
            ))
    except Exception:
        pass

    return arts

def lineplot_clustering(out: dict, split: Split) -> Iterable[PlotArtifact]:
    """
    Plot one time series per octant in 3D feature space.
    For each octant, the time series whose feature vector is closest to the octant center is plotted.
    Additionally, the selected feature values from X_sel (rounded) are shown inside the plot.
    """

    arts: list[PlotArtifact] = []

    # feature matrix for 3D positioning
    df_X = out["splits"][split]["X_train"]
    # selected features to display (around 8 features)
    X_sel = out["splits"][split]["X_sel"]
    # time series values: columns are ts_labels, index are time stamps
    df_ts = out["splits"][split]["df"]

    if hasattr(df_X, "iloc"):
        X = df_X.iloc[:, :3].copy()
        ts_labels = df_X.index
    else:
        X = np.asarray(df_X)[:, :3]
        ts_labels = None

    if hasattr(df_X, "iloc"):
        x_vals = df_X.iloc[:, 0].values
        y_vals = df_X.iloc[:, 1].values
        z_vals = df_X.iloc[:, 2].values
    else:
        x_vals = X[:, 0]
        y_vals = X[:, 1]
        z_vals = X[:, 2]

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    z_min, z_max = np.min(z_vals), np.max(z_vals)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    octants = []
    for xi, (xl, xh) in enumerate([(x_min, x_mid), (x_mid, x_max)]):
        for yi, (yl, yh) in enumerate([(y_min, y_mid), (y_mid, y_max)]):
            for zi, (zl, zh) in enumerate([(z_min, z_mid), (z_mid, z_max)]):
                name = f"PC1-{xi}_PC2-{yi}_PC3-{zi}"
                octants.append((name, (xl, xh), (yl, yh), (zl, zh)))

    for name, (xl, xh), (yl, yh), (zl, zh) in octants:
        in_oct = (
            (x_vals >= xl) & (x_vals <= xh) &
            (y_vals >= yl) & (y_vals <= yh) &
            (z_vals >= zl) & (z_vals <= zh)
        )
        idx_oct = np.where(in_oct)[0]

        if idx_oct.size == 0:
            continue

        cx = 0.5 * (xl + xh)
        cy = 0.5 * (yl + yh)
        cz = 0.5 * (zl + zh)
        center = np.array([cx, cy, cz])

        pts = np.column_stack([x_vals[idx_oct], y_vals[idx_oct], z_vals[idx_oct]])
        dists = np.linalg.norm(pts - center, axis=1)
        best_local_idx = idx_oct[np.argmin(dists)]

        if ts_labels is not None:
            ts_label = ts_labels[best_local_idx]
        else:
            ts_label = best_local_idx

        if ts_label not in df_ts.columns:
            ts_label_str = str(ts_label)
            if ts_label_str in df_ts.columns:
                ts_label = ts_label_str
            else:
                continue

        ts_values = df_ts[ts_label]

        fig, ax = plt.subplots()
        ax.plot(ts_values.index, ts_values.values)
        ax.set_title(f"Time series in octant {name}")
        ax.set_xlabel("t")
        ax.set_ylabel("y(t)")

        # add small feature text from X_sel
        try:
            # align X_sel to ts_label
            if ts_label in X_sel.index:
                sel_row = X_sel.loc[ts_label]
            elif str(ts_label) in X_sel.index:
                sel_row = X_sel.loc[str(ts_label)]
            else:
                sel_row = None

            if sel_row is not None:
                sel_row = sel_row.round(1)
                # start near top right inside axes
                y_pos = 0.98
                for feat_name, feat_val in sel_row.items():
                    ax.text(
                        0.99,
                        y_pos,
                        f"{feat_name}: {feat_val}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6,
                    )
                    y_pos -= 0.07  # move down a bit for next line
        except Exception:
            pass

        arts.append(PlotArtifact(
            id=f"{split}_line_{name}".replace(" ", "_"),
            title=f"Time series in octant {name}",
            kind="line/timeseries",
            split=split,
            fig=fig,
            lib="mpl",
        ))

    return arts