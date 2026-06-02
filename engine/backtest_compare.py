"""
Run ORB backtest with three tuning variants and print side-by-side stats.

Variants:
  A — Loose stop (OR midpoint) + entry buffer + time filter (9:45-11:00)
  B — Variant A + tighter target (0.7x OR_width)
  C — Variant A + OR-drive direction filter (only trade in direction of OR's
      first/last bar tilt)

All three share data (one Kite pull) so the comparison is fair and credit-cheap.

Usage:
  python -m engine.backtest_compare --days 30 --iv 0.13
"""
from __future__ import annotations

import argparse
import sys

from .backtest import run_backtest, summarize, fetch_bars_cached


VARIANTS: dict[str, dict] = {
    "A_loose_stop_buffer_timewin": {
        "stop_mode": "or_midpoint",
        "entry_buffer_frac": 0.15,
        "time_window_start": 30,
        "time_window_end": 105,   # 9:45 + 105 = 11:00
        "direction_filter": "none",
        "target_multiple": 1.0,
    },
    "B_A_plus_tighter_target": {
        "stop_mode": "or_midpoint",
        "entry_buffer_frac": 0.15,
        "time_window_start": 30,
        "time_window_end": 105,
        "direction_filter": "none",
        "target_multiple": 0.7,
    },
    "C_A_plus_or_drive_filter": {
        "stop_mode": "or_midpoint",
        "entry_buffer_frac": 0.15,
        "time_window_start": 30,
        "time_window_end": 105,
        "direction_filter": "or_drive",
        "target_multiple": 1.0,
    },
    "ORIGINAL_for_reference": {
        # The original (failing) configuration — kept as control
        "stop_mode": "break_level",
        "entry_buffer_frac": 0.0,
        "time_window_start": 30,
        "time_window_end": 360,
        "direction_filter": "none",
        "target_multiple": 1.0,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--iv", type=float, default=0.13)
    p.add_argument("--dte", type=int, default=6)
    p.add_argument("--interval", default="5minute",
                   help="Bar interval: 'minute', '3minute', '4minute' (resampled), '5minute'")
    args = p.parse_args()

    bars = fetch_bars_cached(args.days, interval=args.interval)

    results: dict[str, dict] = {}
    for name, kwargs in VARIANTS.items():
        trades = run_backtest(
            days=args.days, iv=args.iv, dte_days=args.dte,
            min_volume_z=-99.0,
            rule_kwargs=kwargs,
            _bars_cache=bars,
        )
        results[name] = {"stats": summarize(trades), "trades": trades, "kwargs": kwargs}

    # Print side-by-side
    print(f"\n=== ORB tuning comparison — {args.days}d @ IV={args.iv*100:.1f}% ===\n")

    # Compact metric table
    metrics = [
        ("trades", "Trades", "{:>4}"),
        ("wins_target", "Target hits", "{:>4}"),
        ("stops", "Stops", "{:>4}"),
        ("timeouts", "Timeouts", "{:>4}"),
        ("win_rate_target", "Target hit %", "{:>5.1%}"),
        ("profit_rate_any", "Any-profit %", "{:>5.1%}"),
        ("avg_win_inr", "Avg win Rs", "{:>+7.0f}"),
        ("avg_loss_inr", "Avg loss Rs", "{:>+7.0f}"),
        ("avg_rr", "R/R |w/l|", "{:>5.2f}"),
        ("total_pnl_inr", "Total Rs", "{:>+9.0f}"),
        ("max_drawdown_inr", "Max DD Rs", "{:>+9.0f}"),
        ("best_trade", "Best", "{:>+7.0f}"),
        ("worst_trade", "Worst", "{:>+7.0f}"),
    ]

    name_w = max(len(n) for n in results) + 2
    headers = list(results.keys())
    # Column widths
    print(f"{'Metric':<20}", end="")
    for h in headers:
        print(f"{h:>30}", end="")
    print()
    print("-" * (20 + 30 * len(headers)))

    def safe_fmt(val, fmt):
        if val is None:
            return "—"
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    for key, label, fmt in metrics:
        print(f"{label:<20}", end="")
        for h in headers:
            v = results[h]["stats"].get(key)
            print(f"{safe_fmt(v, fmt):>30}", end="")
        print()

    print("\nVariant configurations:")
    for h in headers:
        cfg = results[h]["kwargs"]
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {h}: {cfg_str}")

    # Winner
    pnls = {h: results[h]["stats"].get("total_pnl_inr", 0) for h in headers}
    best = max(pnls, key=pnls.get)
    print(f"\nBest by total P&L: {best}  (Rs {pnls[best]:+,.0f})")
    print("\nHonest read: even the best variant must beat zero by enough to cover")
    print("slippage (~10-20% of gross P&L) before paper-trading. Look at R/R and")
    print("number of trades too — 5 trades with great stats are noise.")


if __name__ == "__main__":
    main()
