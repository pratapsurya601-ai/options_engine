"""
CLI for the recommender.

Examples:
  # Live chain, bullish view, 70% target win-rate, ₹50k budget
  python -m engine.cli --view bullish --conviction 0.6 --win 0.7 --capital 50000

  # Use DB-cached chain
  python -m engine.cli --source db --symbol NIFTY --view bearish --win 0.6

  # Include FII/DII bias from latest DB row
  python -m engine.cli --view neutral --win 0.65 --smart-money

  # Override smart-money score manually (no DB)
  python -m engine.cli --view bullish --win 0.7 --sm-score 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

from .chain import chain_from_live, chain_from_db, t_years_to_expiry
from .recommender import recommend, _atm_iv
from .smart_money import score_latest, score_rolling, SmartMoneySignal
from .events import assess as assess_events
from .iv_rank import iv_rank_atm, IVRankResult


IST = timezone(timedelta(hours=5, minutes=30))


def fetch_smart_money(rolling: int = 5) -> SmartMoneySignal | None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        print("psycopg not installed — skipping smart-money", file=sys.stderr)
        return None
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM fii_dii_activity ORDER BY trade_date DESC LIMIT %s",
                (rolling,),
            )
            rows = cur.fetchall()
    if not rows:
        return None
    return score_rolling(rows, days=rolling) if rolling > 1 else score_latest(rows[0])


def main():
    p = argparse.ArgumentParser(description="Options strategy recommender")
    p.add_argument("--source", choices=["live", "db", "manual"], default="live")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--expiry", help="YYYY-MM-DD; default = nearest")
    p.add_argument("--spot", type=float, help="(manual mode) NIFTY spot, e.g. 25000")
    p.add_argument("--iv", type=float, help="(manual mode) ATM IV as decimal, e.g. 0.15")
    p.add_argument("--dte", type=int, default=7, help="(manual mode) days to expiry")
    p.add_argument("--iv-points", type=str, default=None,
                   help='per-strike IV anchors, e.g. "23500:0.135,23900:0.125,24300:0.135"')
    p.add_argument("--calibrate", type=str, default=None,
                   help='strike:type:price triples from chain to back out IV, '
                        'e.g. "23900:CE:188.25,23900:PE:123.85,24300:CE:41.20,23500:PE:23.65"')
    p.add_argument("--view", choices=["bullish", "bearish", "neutral", "vol_long"], required=True)
    p.add_argument("--conviction", type=float, default=0.5, help="0..1")
    p.add_argument("--win", type=float, default=0.6, help="target win-prob threshold (0..1)")
    p.add_argument("--capital", type=float, default=50_000.0, help="max risk budget INR")
    p.add_argument("--lot-size", type=int, default=75)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--smart-money", action="store_true", help="read FII/DII from DB")
    p.add_argument("--sm-rolling", type=int, default=5, help="rolling window days")
    p.add_argument("--sm-score", type=float, help="override smart-money score (-1..+1)")
    p.add_argument("--no-iv-rank", action="store_true", help="skip IV-rank computation")
    p.add_argument("--iv-window", type=int, default=60, help="IV-rank lookback days")
    p.add_argument("--show-rejected", action="store_true",
                   help="also show top near-miss ideas below the win threshold")
    p.add_argument("--scalp", action="store_true",
                   help="scalp mode: rank naked-long CE/PE for short-target premium gains")
    p.add_argument("--target-pts", type=float, default=15.0,
                   help="(scalp) target premium gain in points")
    p.add_argument("--hold-days", type=float, default=1.0,
                   help="(scalp) intended holding window in days (0.25 = intraday)")
    p.add_argument("--max-premium", type=float, default=20_000.0,
                   help="(scalp) max premium per lot in INR")
    p.add_argument("--min-prob-touch", type=float, default=0.40,
                   help="(scalp) minimum P(touch) threshold")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # Build chain
    if args.source == "live":
        chain = chain_from_live(args.symbol)
    elif args.source == "manual":
        if args.spot is None or args.iv is None:
            print("manual mode requires --spot and --iv", file=sys.stderr); sys.exit(2)
        from .manual_chain import synthetic_chain, make_iv_curve, calibrate_iv_from_quotes

        iv_curve = None
        if args.calibrate:
            t_yr = args.dte / 365.0
            quotes = []
            for item in args.calibrate.split(","):
                k_str, ot, px = item.strip().split(":")
                quotes.append((int(k_str), ot.upper(), float(px)))
            anchors = calibrate_iv_from_quotes(args.spot, t_yr, quotes)
            if not anchors:
                print("calibration produced no valid IVs", file=sys.stderr); sys.exit(2)
            iv_curve = make_iv_curve(anchors)
            print(f"Calibrated IV anchors: " +
                  ", ".join(f"{k}={v*100:.1f}%" for k, v in sorted(anchors.items())))
        elif args.iv_points:
            anchors = {}
            for item in args.iv_points.split(","):
                k_str, v_str = item.strip().split(":")
                anchors[int(k_str)] = float(v_str)
            iv_curve = make_iv_curve(anchors)

        chain = synthetic_chain(symbol=args.symbol, spot=args.spot, iv=args.iv,
                                dte_days=args.dte, strike_step=args.step,
                                iv_curve=iv_curve)
    else:
        import psycopg
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("DATABASE_URL not set", file=sys.stderr); sys.exit(2)
        expiry = None
        if args.expiry:
            from datetime import date
            expiry = date.fromisoformat(args.expiry)
        with psycopg.connect(db_url) as conn:
            chain = chain_from_db(conn, args.symbol, expiry)

    t = t_years_to_expiry(chain.expiry)

    # Smart money
    sm_score = 0.0
    sm_sig: SmartMoneySignal | None = None
    if args.sm_score is not None:
        sm_score = args.sm_score
    elif args.smart_money:
        sm_sig = fetch_smart_money(args.sm_rolling)
        if sm_sig:
            sm_score = sm_sig.score

    # Event-risk assessment (always on — pure static data)
    from datetime import date
    today = date.today()
    event_risk = assess_events(today, chain.expiry, iv_rank=None)

    # IV-rank — try DB, gracefully degrade
    iv_rank_info: IVRankResult | None = None
    if not args.no_iv_rank:
        db_url = os.environ.get("DATABASE_URL")
        atm_iv = _atm_iv(chain)
        dte_days = max(int(t * 365), 1)
        if db_url:
            try:
                import psycopg
                with psycopg.connect(db_url) as conn:
                    iv_rank_info = iv_rank_atm(conn, args.symbol, atm_iv, dte_days,
                                                window_days=args.iv_window)
            except Exception as e:
                print(f"IV-rank degraded (DB unreachable: {type(e).__name__})", file=sys.stderr)
        if iv_rank_info is None:
            # Static fallback path
            iv_rank_info = iv_rank_atm.__wrapped__(None, args.symbol, atm_iv, dte_days) \
                if hasattr(iv_rank_atm, "__wrapped__") else IVRankResult(
                    value=None, percentile=None, current_iv=atm_iv,
                    window_days=args.iv_window, sample_count=0,
                    confidence="insufficient",
                    bucket="normal" if 0.13 <= atm_iv <= 0.20 else ("cheap" if atm_iv < 0.13 else "rich"),
                    advice="No DB — using static IV buckets only",
                )

        # Re-assess events now that we know IV-rank (for the 'IV cheap + event' carve-out)
        event_risk = assess_events(today, chain.expiry, iv_rank=iv_rank_info.value)

    # ---------------- SCALP MODE BRANCH ----------------
    if args.scalp:
        from .scalp import scalp_recommend
        scalp_ideas = scalp_recommend(
            chain=chain, t_years_to_expiry=t,
            view=args.view, conviction=args.conviction,
            target_points=args.target_pts, hold_days=args.hold_days,
            max_premium_per_lot=args.max_premium,
            capital_inr=args.capital, lot_size=args.lot_size,
            min_prob_touch=args.min_prob_touch, top_n=args.top,
        )
        print(f"\n=== {chain.symbol} @ {chain.spot:.2f}  expiry {chain.expiry}  T={t*365:.1f}d ===")
        print(f"SCALP mode: target +{args.target_pts:.0f} pts of premium  within {args.hold_days:.1f}d  view: {args.view}  conviction: {args.conviction:.2f}")
        print(f"Budget: Rs {args.capital:,.0f}  max-premium-per-lot: Rs {args.max_premium:,.0f}  min P(touch): {args.min_prob_touch*100:.0f}%\n")
        if not scalp_ideas:
            print("No candidates clear your P(touch) threshold.")
            print("Try: lower --min-prob-touch, raise --hold-days, or relax --target-pts.")
            return
        for i, idea in enumerate(scalp_ideas, 1):
            s = idea.summary()
            lots_affordable = int(args.capital // s['cost_per_lot'])
            print(f"[{i}] BUY {s['type']} {s['strike']} @ Rs {s['premium']:.2f}  (lots fitting budget: {lots_affordable})")
            print(f"    P(touch)={s['prob_touch']*100:.1f}%   target +Rs {s['target_gain_per_lot']:,.0f}/lot   bleed if missed: -Rs {s['theta_bleed_per_lot']:,.0f}/lot")
            print(f"    cost/lot: Rs {s['cost_per_lot']:,.0f}   max loss/lot: -Rs {s['max_loss_per_lot']:,.0f}   E[P&L]/lot: Rs {s['expected_pnl_per_lot']:,.0f}   R/R: {s['risk_reward']:.2f}")
            for r in s['rationale']:
                print(f"    - {r}")
            print()
        return

    result = recommend(
        chain=chain, t_years=t,
        view=args.view, conviction=args.conviction,
        target_win_pct=args.win, max_capital_inr=args.capital,
        smart_money_score=sm_score,
        lot_size=args.lot_size, strike_step=args.step,
        top_n=args.top,
        event_risk=event_risk,
        iv_rank_info=iv_rank_info,
        include_rejected=args.show_rejected,
    )
    if args.show_rejected:
        ideas, rejected = result
    else:
        ideas, rejected = result, []

    if args.json:
        out = {
            "symbol": chain.symbol, "spot": chain.spot,
            "expiry": chain.expiry.isoformat(),
            "t_days": round(t * 365, 2),
            "smart_money": {"score": sm_score, "label": sm_sig.label if sm_sig else None},
            "event_risk": {
                "spans_event": event_risk.spans_event,
                "events": [{"name": e.name, "date": e.event_date.isoformat()} for e in event_risk.events],
                "pre_event_now": [{"name": e.name, "date": e.event_date.isoformat()} for e in event_risk.pre_event_now],
                "advice": event_risk.advice,
            } if event_risk else None,
            "iv_rank": {
                "value": iv_rank_info.value,
                "percentile": iv_rank_info.percentile,
                "current_iv": iv_rank_info.current_iv,
                "bucket": iv_rank_info.bucket,
                "confidence": iv_rank_info.confidence,
                "sample_count": iv_rank_info.sample_count,
                "advice": iv_rank_info.advice,
            } if iv_rank_info else None,
            "ideas": [i.summary() for i in ideas],
        }
        print(json.dumps(out, indent=2, default=str))
        return

    # Human-readable
    print(f"\n=== {chain.symbol} @ {chain.spot:.2f}  expiry {chain.expiry}  T={t*365:.1f}d ===")
    print(f"View: {args.view}  conviction: {args.conviction:.2f}  target win: {args.win*100:.0f}%  budget: Rs {args.capital:,.0f}")
    if sm_sig:
        print(f"Smart money: {sm_sig.label}  score={sm_sig.score:+.2f}  ({', '.join(sm_sig.notes) or 'ok'})")
    elif args.sm_score is not None:
        print(f"Smart money (manual): {sm_score:+.2f}")
    if iv_rank_info:
        rk = f"{iv_rank_info.value*100:.0f}%" if iv_rank_info.value is not None else "n/a"
        print(f"IV-rank: {rk}  bucket={iv_rank_info.bucket}  confidence={iv_rank_info.confidence}  (n={iv_rank_info.sample_count}d)")
        if iv_rank_info.advice:
            print(f"  -> {iv_rank_info.advice}")
    if event_risk and (event_risk.spans_event or event_risk.pre_event_now):
        for line in event_risk.to_rationale():
            print(f"EVENT: {line}")
    print()

    if not ideas:
        print("No trades meet your target win-rate. Lower --win, raise --conviction, or extend --dte.")
        if not args.show_rejected:
            print("Re-run with --show-rejected to see the best near-misses and their actual win-rates.")
            return

    def _print_idea(label, idea):
        s = idea.summary()
        print(f"{label} {s['name']}  ({s['view']})  lots={s['lots']}")
        legs = "  ".join(
            f"{l['action']} {l['type']} {l['strike']} @ Rs {l['premium']:.2f}"
            for l in s['legs']
        )
        print(f"    legs: {legs}")
        print(f"    cost: Rs {s['cost_inr']:,.0f}   max P: Rs {s['max_profit_inr']:,.0f}   max L: Rs {s['max_loss_inr']:,.0f}")
        print(f"    BE: {s['breakevens']}   P(win)={s['prob_profit']*100:.1f}%   E[P&L]=Rs {s['expected_payoff_inr']:,.0f}   R/R={s['return_on_risk']:.2f}")
        for r in s['rationale']:
            print(f"    - {r}")
        print()

    for i, idea in enumerate(ideas, 1):
        _print_idea(f"[{i}]", idea)

    if rejected:
        print("--- NEAR-MISSES (below your win threshold) ---\n")
        for i, idea in enumerate(rejected, 1):
            _print_idea(f"(x{i})", idea)


if __name__ == "__main__":
    main()
