"""
Multi-timeframe comparison for a single rule.

Pulls 1-min NIFTY bars ONCE, resamples to multiple intervals, runs the same
rule against each, and prints side-by-side stats. Tells you whether timeframe
materially changes a rule's performance — or whether the regime overrides TF.

Usage:
  python -m engine.backtest_tf_compare --rule ema_sar --days 30 --iv 0.13
"""
from __future__ import annotations

import argparse
import sys

from .backtest import run_backtest, summarize, fetch_bars_cached, _resample_bars


INTERVALS = [
    ("2min", 2),
    ("3min", 3),
    ("5min", 5),
    ("10min", 10),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rule", default="ema_sar",
                   help="orb | macd_ema_oi | ema_sar")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--iv", type=float, default=0.13)
    p.add_argument("--dte", type=int, default=6)
    args = p.parse_args()

    # One Kite pull at 1-minute resolution
    bars_1m = fetch_bars_cached(args.days, interval="minute")

    results = {}
    for label, group in INTERVALS:
        bars = _resample_bars(bars_1m, group) if group > 1 else bars_1m
        print(f"\nRunning {args.rule} on {label} bars ({len(bars)} bars)...",
              file=sys.stderr)
        trades = run_backtest(
            days=args.days, iv=args.iv, dte_days=args.dte,
            rule_name=args.rule, _bars_cache=bars,
        )
        results[label] = {"stats": summarize(trades), "trades": trades}

    # Side-by-side table
    print(f"\n=== {args.rule.upper()} across timeframes — {args.days}d @ IV={args.iv*100:.1f}% ===\n")

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

    headers = list(results.keys())
    print(f"{'Metric':<18}", end="")
    for h in headers:
        print(f"{h:>14}", end="")
    print()
    print("-" * (18 + 14 * len(headers)))

    def fmt_val(v, fmt):
        if v is None:
            return "—"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    for key, label, fmt in metrics:
        print(f"{label:<18}", end="")
        for h in headers:
            v = results[h]["stats"].get(key)
            print(f"{fmt_val(v, fmt):>14}", end="")
        print()

    pnls = {h: results[h]["stats"].get("total_pnl_inr", 0) for h in headers}
    best = max(pnls, key=pnls.get) if pnls else None
    if best:
        print(f"\nBest TF by total P&L: {best}  (Rs {pnls[best]:+,.0f})")
    print("\nIf all TFs lose by similar amounts, timeframe is NOT the constraint.")
    print("That's evidence the rule's premise (continuation) is misaligned with regime.")


if __name__ == "__main__":
    main()
