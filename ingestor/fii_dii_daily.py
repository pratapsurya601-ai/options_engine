"""
Daily FII/DII ingestion runner.

Designed for GitHub Actions cron @ 7 PM IST. Pulls NSE's cash + derivatives
FII/DII numbers, normalizes, writes into the fii_dii_activity table.

Best-effort: failures DON'T fail the workflow (we'll retry tomorrow).

Run:
  python -m ingestor.fii_dii_daily

Env vars:
  DATABASE_URL   - Aiven Postgres (required)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, date, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger(__name__)


def _summarize_cash(rows: list[dict]) -> dict | None:
    """Reduce NSE cash response into one canonical row for today.
    NSE's response shape changes; defend against missing keys."""
    if not rows:
        return None
    # Pick row tagged with the latest date
    rows = sorted(rows, key=lambda r: str(r.get("date") or r.get("reportDate") or ""), reverse=True)
    fii = {"buy": 0.0, "sell": 0.0, "net": 0.0}
    dii = {"buy": 0.0, "sell": 0.0, "net": 0.0}
    trade_date = None
    for r in rows:
        category = str(r.get("category") or "").upper()
        try:
            buy = float(r.get("buyValue") or r.get("grossPurchase") or 0)
            sell = float(r.get("sellValue") or r.get("grossSales") or 0)
            net = float(r.get("netValue") or r.get("netInvestment") or buy - sell)
        except (TypeError, ValueError):
            continue
        if "FII" in category or "FPI" in category:
            fii = {"buy": buy, "sell": sell, "net": net}
        elif "DII" in category:
            dii = {"buy": buy, "sell": sell, "net": net}
        if trade_date is None:
            d = r.get("date") or r.get("reportDate")
            if d:
                trade_date = str(d)
    return {
        "fii_cash_buy": fii["buy"],
        "fii_cash_sell": fii["sell"],
        "fii_cash_net": fii["net"],
        "dii_cash_buy": dii["buy"],
        "dii_cash_sell": dii["sell"],
        "dii_cash_net": dii["net"],
        "trade_date_str": trade_date,
    }


def _summarize_deriv(rows: list[dict]) -> dict:
    """Extract latest day's FII index futures net positioning.
    Field names are erratic; pull both common variants."""
    out = {
        "fii_index_fut_long_contracts": 0,
        "fii_index_fut_short_contracts": 0,
        "fii_index_fut_net_contracts": 0,
        "fii_index_fut_long_short_ratio": None,
        "fii_index_call_long": 0,
        "fii_index_call_short": 0,
        "fii_index_put_long": 0,
        "fii_index_put_short": 0,
        "fii_stock_fut_long": 0,
        "fii_stock_fut_short": 0,
    }
    if not rows:
        return out
    latest = sorted(rows, key=lambda r: str(r.get("date") or ""), reverse=True)[0]

    def _ig(name, default=0):
        v = latest.get(name)
        try:
            return int(float(v)) if v is not None else default
        except (TypeError, ValueError):
            return default

    out["fii_index_fut_long_contracts"]  = _ig("indexFuturesLongContract")  or _ig("indexFutLong")
    out["fii_index_fut_short_contracts"] = _ig("indexFuturesShortContract") or _ig("indexFutShort")
    out["fii_index_call_long"]           = _ig("indexCallOptionsLongContract") or _ig("indexCallLong")
    out["fii_index_call_short"]          = _ig("indexCallOptionsShortContract") or _ig("indexCallShort")
    out["fii_index_put_long"]            = _ig("indexPutOptionsLongContract") or _ig("indexPutLong")
    out["fii_index_put_short"]           = _ig("indexPutOptionsShortContract") or _ig("indexPutShort")
    out["fii_stock_fut_long"]            = _ig("stockFuturesLongContract") or _ig("stockFutLong")
    out["fii_stock_fut_short"]           = _ig("stockFuturesShortContract") or _ig("stockFutShort")
    out["fii_index_fut_net_contracts"]   = (out["fii_index_fut_long_contracts"]
                                             - out["fii_index_fut_short_contracts"])
    if out["fii_index_fut_short_contracts"]:
        out["fii_index_fut_long_short_ratio"] = round(
            out["fii_index_fut_long_contracts"] / out["fii_index_fut_short_contracts"], 4)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)

    today = datetime.now(tz=IST).date()
    print(f"[{datetime.now(tz=IST).strftime('%H:%M:%S')}] FII/DII ingest start for {today}")

    # Lazy import — keeps unit tests fast
    from .nse_fetcher import NSEClient
    from .fii_dii_fetcher import fetch_fii_dii_cash, fetch_fii_derivatives
    from db.writer import get_conn, write_fii_dii

    nse = NSEClient(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        timeout=15,
    )
    cash_rows = []
    deriv_rows = []

    try:
        cash_rows = fetch_fii_dii_cash(nse) or []
        print(f"cash rows: {len(cash_rows)}")
    except Exception as e:
        print(f"cash fetch failed: {e}", file=sys.stderr)

    try:
        deriv_rows = fetch_fii_derivatives(nse) or []
        print(f"deriv rows: {len(deriv_rows)}")
    except Exception as e:
        print(f"deriv fetch failed: {e}", file=sys.stderr)

    if not cash_rows and not deriv_rows:
        print("Both fetches empty — nothing to write. Exiting 0 (will retry tomorrow).")
        sys.exit(0)

    cash_summary = _summarize_cash(cash_rows) or {}
    deriv_summary = _summarize_deriv(deriv_rows)

    row = {
        "trade_date": today,
        "fii_cash_buy":  cash_summary.get("fii_cash_buy")  or 0,
        "fii_cash_sell": cash_summary.get("fii_cash_sell") or 0,
        "fii_cash_net":  cash_summary.get("fii_cash_net")  or 0,
        "dii_cash_buy":  cash_summary.get("dii_cash_buy")  or 0,
        "dii_cash_sell": cash_summary.get("dii_cash_sell") or 0,
        "dii_cash_net":  cash_summary.get("dii_cash_net")  or 0,
        **deriv_summary,
        "raw_payload": json.dumps({
            "cash_count": len(cash_rows),
            "deriv_count": len(deriv_rows),
            "ingested_at_ist": datetime.now(tz=IST).isoformat(),
            "cash_sample": cash_rows[:3],
            "deriv_sample": deriv_rows[:1],
        }),
    }

    try:
        with get_conn(db_url) as conn:
            write_fii_dii(conn, row)
        print(f"OK: fii_cash_net={row['fii_cash_net']:.0f} "
              f"dii_cash_net={row['dii_cash_net']:.0f} "
              f"fii_idx_fut_net={row['fii_index_fut_net_contracts']:,} "
              f"fii_idx_fut_ratio={row['fii_index_fut_long_short_ratio']}")
    except Exception as e:
        print(f"DB write failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
