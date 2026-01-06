import numpy as np
import pandas as pd

import pandas as pd

def cut_last_non_na(
    df: pd.DataFrame,
    cut_len: int,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Set the last `cut_len` non-NaN values in each column of `df` to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cut_len : int
        Number of last non-NaN values per column to set to NaN.
    inplace : bool, optional
        If True, modify `df` in place and also return it.
        If False (default), work on a copy and return the modified copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with the last `cut_len` non-NaN values per column set to NaN.
    """
    if cut_len <= 0:
        return df if inplace else df.copy()

    if not inplace:
        df = df.copy()

    for col in df.columns:
        non_na_idx = df.index[df[col].notna()]
        if len(non_na_idx) == 0:
            continue

        tail_idx = non_na_idx[-cut_len:]
        df.loc[tail_idx, col] = pd.NA

    return df


def split_windows(
    df: pd.DataFrame,
    n: int,
    fc_horizon: int,
    pct: float = 0.80,
    sep: str = "_",
    trim_edge_nans: bool = True,
) -> pd.DataFrame:
    """
    Split a time series DataFrame into n overlapping windows.

    If trim_edge_nans=True, leading and trailing rows where *all* columns are NaN
    are removed before windowing. This makes it safe to call the function again
    on a DataFrame that already contains edge-NaNs from previous processing.
    """

    if n <= 0:
        raise ValueError("`n` must be >= 1.")
    if fc_horizon < 0:
        raise ValueError("`fc_horizon` must be >= 0.")

    df_work = df

    if trim_edge_nans:
        mask = df_work.notna().any(axis=1)
        if not mask.any():
            return pd.DataFrame()

        first_idx = mask.idxmax()
        last_idx = mask[::-1].idxmax()

        df_work = df_work.loc[first_idx:last_idx]

    T = int(len(df_work))
    if T == 0:
        return pd.DataFrame()

    H = int(fc_horizon)
    if T <= H:
        return pd.DataFrame()

    L_target = H + int(round(pct * T))
    L = max(H + 1, min(T, L_target))

    max_n = max(1, T - L + 1)
    n_eff = min(int(n), max_n)

    blocks = []
    for i in range(n_eff):
        start = i
        end = start + L
        win = df_work.iloc[start:end].copy()

        win = win.ffill().bfill()
        win = win.loc[:, ~win.isna().all(axis=0)]

        win = win.add_suffix(f"{sep}{i+1}")
        blocks.append(win)

    if not blocks:
        return pd.DataFrame()

    return pd.concat(blocks, axis=1)

def split_df_train_val_test(
    df: pd.DataFrame,
    n: int = 5,
    fc_horizon: int = 26,
    step_unit: int = 52,
    base_blocks: int = 4,
):
    """
    Build rolling-like train/val/test windows from a single time series
    such that:
      - windows themselves may overlap (within train, within val, within test),
      - but the final fc_horizon points (the backtest target part) of train,
        val and test do not overlap in time.
    The function tries to keep the window length as large as possible,
    starting from base_blocks * step_unit, and reduces it in step_unit
    increments if there is not enough room.
    """
    T = len(df)
    if T == 0:
        raise ValueError("Empty dataframe.")

    # start from the largest candidate window length
    ts_length = min(base_blocks * step_unit, T)

    chosen_len = None
    chosen_step = None

    # search for a window length and an inner step that fit into the series
    # layout idea:
    #   [ ... train windows ... ][ ... val windows ... ][ ... test windows ... ]
    # test windows are at the very end, val is fc_horizon before the earliest test,
    # train is fc_horizon before the earliest val
    while ts_length > 0 and chosen_len is None:
        # remaining space after one window and two safety gaps train->val, val->test
        remaining = T - ts_length - 2 * fc_horizon
        if remaining <= 0:
            ts_length -= step_unit
            continue

        if n == 1:
            # if we only want one window per split, we can always step by 1
            chosen_len = ts_length
            chosen_step = 1
            break

        # we need room to shift n windows in each of the 3 splits
        # with step = s this costs 3 * (n-1) * s
        max_step = remaining // (3 * (n - 1))
        if max_step >= 1:
            chosen_len = ts_length
            chosen_step = max_step
            break

        ts_length -= step_unit

    if chosen_len is None:
        raise ValueError("Time series too short for the requested setup.")

    ts_length = chosen_len
    step = chosen_step

    # build end indices (exclusive) for test windows
    test_ends = [T - i * step for i in range(n)]
    earliest_test_end = test_ends[-1]

    # val windows must end at least fc_horizon earlier than the earliest test window
    val_first_end = earliest_test_end - fc_horizon
    val_ends = [val_first_end - i * step for i in range(n)]
    earliest_val_end = val_ends[-1]

    # train windows must end at least fc_horizon earlier than the earliest val window
    train_first_end = earliest_val_end - fc_horizon
    train_ends = [train_first_end - i * step for i in range(n)]

    def make_window(end_idx: int, length: int) -> pd.DataFrame:
        start_idx = end_idx - length
        win = df.iloc[start_idx:end_idx].copy() # .reset_index(drop=True)
        # forward/backward fill to avoid NaNs if there are any
        return win.ffill().bfill()

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    for e in train_ends:
        df_train = pd.concat([df_train, make_window(e, ts_length)], axis=1, ignore_index=True)
    for e in val_ends:
        df_val = pd.concat([df_val, make_window(e, ts_length)], axis=1, ignore_index=True)
    for e in test_ends:
        df_test = pd.concat([df_test, make_window(e, ts_length)], axis=1, ignore_index=True)

    for dfx in (df_train, df_val, df_test):
        dfx.columns = [f"ts{i+1}" for i in range(dfx.shape[1])]

    return df_train, df_val, df_test

