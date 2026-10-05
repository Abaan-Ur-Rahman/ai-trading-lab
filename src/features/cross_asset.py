"""Cross-asset features derived from a second, correlated instrument.

These exist because the current feature set (EMA/RSI/ATR/MACD/returns, all
derived from XAUUSD's own OHLCV) was shown, via a binary direction test and
a horizon sweep documented in the README, to carry only a small (~2-4
percentage point) directional edge that does not improve with more
tuning. A second, correlated instrument (e.g. XAG/USD) is genuinely new
information the single-symbol feature set cannot see at all -- not
another transform of the same price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_secondary_close(
    primary_index: pd.DatetimeIndex,
    secondary_close: pd.Series,
) -> pd.Series:
    """Align a secondary instrument's close prices onto the primary index.

    Forward-fills gaps (e.g. the secondary symbol missing a bar the
    primary has) using only that secondary instrument's own past values --
    never looks ahead. Timestamps in `primary_index` before the secondary
    series' first observation stay NaN, since there is no past secondary
    value yet.
    """
    return secondary_close.reindex(primary_index, method="ffill")


def synthetic_dollar_index(
    eurusd_close: pd.Series,
    usdjpy_close: pd.Series,
) -> pd.Series:
    """Equal-weighted dollar-strength index from EUR/USD and USD/JPY.

    Both pairs are quoted differently: the dollar strengthening pushes
    EUR/USD down but USD/JPY up. So the index is built from the dollar side
    of each, in log space:

        log(index) = 0.5 * (log(USD/JPY) - log(EUR/USD))

    It rises when the dollar strengthens against both currencies. Equal
    weights are deliberate: with only two pairs, the real DXY's weights
    (EUR ~58%, JPY ~14%) would make this nearly identical to EUR/USD alone,
    defeating the point of averaging out each pair's own noise. Only the
    index's moves matter (every feature built from it uses log returns or
    ratios of returns), so its absolute level is arbitrary.

    The two series are aligned onto the union of their timestamps by
    forward-filling each one's own past values (causal, never looks ahead).
    Leading timestamps before both series have started are dropped.
    """
    union_index = eurusd_close.index.union(usdjpy_close.index).sort_values()
    eurusd = eurusd_close.reindex(union_index, method="ffill")
    usdjpy = usdjpy_close.reindex(union_index, method="ffill")

    log_index = 0.5 * (np.log(usdjpy) - np.log(eurusd))

    return np.exp(log_index).dropna().rename("close")


def build_cross_asset_features(
    primary_close: pd.Series,
    secondary_close: pd.Series,
    corr_window: int = 20,
) -> pd.DataFrame:
    """Build features comparing `primary_close` to a second instrument.

    `secondary_close` is first aligned onto `primary_close`'s index
    (forward-filled, causal -- see align_secondary_close). Returns three
    columns, each using only current/past information at every row:

    - secondary_log_return: the second instrument's own 1-bar log return.
    - ratio_log_return: 1-bar log change in the primary/secondary price
      ratio (e.g. the gold/silver ratio) -- captures relative moves
      between the two instruments, not just each one's own direction.
    - rolling_correlation: trailing `corr_window`-bar correlation between
      the two instruments' 1-bar log returns -- whether they are
      currently moving together or decoupling.

    Leading rows without enough history are NaN, the same warm-up
    convention as feature_builder.py. Does not mutate either input Series.
    """
    if corr_window <= 1:
        raise ValueError("corr_window must be greater than 1")

    aligned_secondary = align_secondary_close(primary_close.index, secondary_close)

    primary_log_return = np.log(primary_close / primary_close.shift(1))
    secondary_log_return = np.log(aligned_secondary / aligned_secondary.shift(1))

    ratio = primary_close / aligned_secondary
    ratio_log_return = np.log(ratio / ratio.shift(1))

    rolling_correlation = primary_log_return.rolling(window=corr_window).corr(secondary_log_return)

    # A window where either instrument did not move at all (common for the
    # secondary: forward-filled through its own data gaps while the primary
    # keeps trading) has no defined correlation. pandas' rolling corr leaves
    # ~1e-11 of floating-point residue in such windows, so whether it returns
    # NaN or an arbitrary value depends on the platform's floating-point
    # rounding -- the same data produced different datasets on Linux and Windows.
    # Flat windows are set explicitly to 0.0 ("no measurable co-movement"),
    # which is platform-independent and keeps the dataset contiguous (the
    # positional purge in ml.dataset.chronological_split assumes no internal
    # gaps). Genuine 1h return std is orders of magnitude above the tolerance.
    flat_std_tolerance = 1e-9
    primary_flat = primary_log_return.rolling(window=corr_window).std() < flat_std_tolerance
    secondary_flat = secondary_log_return.rolling(window=corr_window).std() < flat_std_tolerance
    has_full_window = primary_log_return.rolling(window=corr_window).count().eq(corr_window) & (
        secondary_log_return.rolling(window=corr_window).count().eq(corr_window)
    )
    rolling_correlation = rolling_correlation.where(~(has_full_window & (primary_flat | secondary_flat)), 0.0)
    rolling_correlation = rolling_correlation.clip(-1.0, 1.0)

    return pd.DataFrame(
        {
            "secondary_log_return": secondary_log_return,
            "ratio_log_return": ratio_log_return,
            "rolling_correlation": rolling_correlation,
        },
        index=primary_close.index,
    )