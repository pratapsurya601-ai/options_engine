"""
Technical indicators — pure functions over OHLC bar sequences.

All functions return a list of values aligned with input bars (None for warmup).
"""
from __future__ import annotations

from typing import Sequence


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average. Returns None for first `period-1` values."""
    if period <= 0:
        raise ValueError("period must be >0")
    if len(values) < period:
        return [None] * len(values)
    k = 2.0 / (period + 1)
    out: list[float | None] = [None] * (period - 1)
    # Seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    out.append(seed)
    prev = seed
    for v in values[period:]:
        ema_v = v * k + prev * (1 - k)
        out.append(ema_v)
        prev = ema_v
    return out


def macd(closes: Sequence[float], fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """
    Returns (macd_line, signal_line, histogram).
    macd_line = EMA(fast) - EMA(slow); signal_line = EMA(macd_line, signal).
    """
    ef = ema(closes, fast)
    es = ema(closes, slow)
    macd_line: list[float | None] = []
    for a, b in zip(ef, es):
        if a is None or b is None:
            macd_line.append(None)
        else:
            macd_line.append(a - b)
    # Signal = EMA of macd_line (only over non-None entries)
    first_idx = next((i for i, v in enumerate(macd_line) if v is not None), None)
    if first_idx is None:
        return macd_line, [None]*len(closes), [None]*len(closes)
    valid = [v for v in macd_line if v is not None]
    sig_valid = ema(valid, signal)
    signal_line: list[float | None] = [None] * first_idx + sig_valid
    while len(signal_line) < len(closes):
        signal_line.append(None)
    hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def bollinger(closes: Sequence[float], period: int = 20,
              num_std: float = 2.0) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return (middle, upper, lower) bands. Middle = SMA(period). Upper/Lower = middle +/- num_std * stdev.

    Aligned with input; first (period-1) entries are None (warmup).
    """
    if period <= 0:
        raise ValueError("period must be >0")
    n = len(closes)
    middle: list[float | None] = [None] * n
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    if n < period:
        return middle, upper, lower
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        middle[i] = m
        upper[i] = m + num_std * sd
        lower[i] = m - num_std * sd
    return middle, upper, lower


def parabolic_sar(highs: Sequence[float], lows: Sequence[float],
                  af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.2) -> list[float | None]:
    """
    Parabolic SAR (Welles Wilder). Returns SAR value per bar; None for bar 0.
    Trend determined by initial bar; flips when price crosses SAR.
    """
    n = len(highs)
    if n < 2:
        return [None] * n
    sar: list[float | None] = [None]
    # Determine initial trend by first two bars
    uptrend = highs[1] > highs[0]
    ep = highs[1] if uptrend else lows[1]    # extreme point
    af = af_start
    sar.append(lows[0] if uptrend else highs[0])

    for i in range(2, n):
        prev_sar = sar[-1]
        new_sar = prev_sar + af * (ep - prev_sar)
        if uptrend:
            # SAR can't be above previous two lows
            new_sar = min(new_sar, lows[i-1], lows[i-2] if i >= 2 else lows[i-1])
            if lows[i] < new_sar:
                # Flip to downtrend
                uptrend = False
                new_sar = ep
                ep = lows[i]
                af = af_start
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = max(new_sar, highs[i-1], highs[i-2] if i >= 2 else highs[i-1])
            if highs[i] > new_sar:
                uptrend = True
                new_sar = ep
                ep = highs[i]
                af = af_start
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)
        sar.append(new_sar)
    return sar


def rsi(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """
    Relative Strength Index using Wilder's smoothing.
    Returns list aligned with input; None for the first `period` warmup values.
    """
    if period <= 0:
        raise ValueError("period must be >0")
    n = len(closes)
    if n <= period:
        return [None] * n

    out: list[float | None] = [None] * (period)
    # Compute initial average gain/loss over first `period` deltas
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / period
    avg_loss = losses / period

    # First RSI value at index `period`
    if avg_loss == 0:
        out.append(100.0 if avg_gain > 0 else 50.0)
    else:
        rs = avg_gain / avg_loss
        out.append(100.0 - (100.0 / (1.0 + rs)))

    # Wilder smoothing for the remainder
    for i in range(period + 1, n):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def crossed_above(series: Sequence[float | None], reference: Sequence[float | None],
                  within_bars: int = 3) -> bool:
    """True if `series` crossed above `reference` within the last `within_bars` bars."""
    n = len(series)
    for i in range(max(1, n - within_bars), n):
        a_prev, a_now = series[i-1], series[i]
        b_prev, b_now = reference[i-1], reference[i]
        if None in (a_prev, a_now, b_prev, b_now):
            continue
        if a_prev <= b_prev and a_now > b_now:
            return True
    return False


def crossed_below(series: Sequence[float | None], reference: Sequence[float | None],
                  within_bars: int = 3) -> bool:
    n = len(series)
    for i in range(max(1, n - within_bars), n):
        a_prev, a_now = series[i-1], series[i]
        b_prev, b_now = reference[i-1], reference[i]
        if None in (a_prev, a_now, b_prev, b_now):
            continue
        if a_prev >= b_prev and a_now < b_now:
            return True
    return False
