"""
FII/DII data ingestor.

NSE publishes two relevant datasets:
1. Cash market FII/DII activity (daily)
2. FII derivatives stats (daily, ~6pm IST)

We pull both and normalize into fii_dii_activity table.
"""
import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

NSE_FII_DII_CASH = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_FII_DERIV = "https://www.nseindia.com/api/fiidaily"


def fetch_fii_dii_cash(nse_client) -> list[dict]:
    """Cash market FII/DII. Returns list (NSE returns multiple recent days)."""
    r = nse_client.session.get(NSE_FII_DII_CASH, timeout=nse_client.timeout)
    r.raise_for_status()
    return r.json()


def fetch_fii_derivatives(nse_client) -> list[dict]:
    """FII derivatives positioning. Index futures L/S is the key signal."""
    r = nse_client.session.get(NSE_FII_DERIV, timeout=nse_client.timeout)
    r.raise_for_status()
    return r.json()


def normalize_cash_row(raw: dict) -> Optional[dict]:
    """NSE field names are inconsistent across endpoints — defend against drift."""
    # Expected keys vary; common ones:
    # category, buyValue, sellValue, netValue, date
    try:
        return {
            "category": raw.get("category", "").upper(),
            "buy": float(raw.get("buyValue") or raw.get("grossPurchase") or 0),
            "sell": float(raw.get("sellValue") or raw.get("grossSales") or 0),
            "net": float(raw.get("netValue") or raw.get("netInvestment") or 0),
            "date": _parse_date(raw.get("date") or raw.get("reportDate")),
        }
    except (ValueError, TypeError) as e:
        logger.warning("Failed to normalize cash row: %s | %s", raw, e)
        return None


def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def build_combined_row(cash_data: list, deriv_data: list, target_date: date) -> Optional[dict]:
    """Merge cash and derivatives data for a single date."""
    cash_rows = [normalize_cash_row(r) for r in cash_data if normalize_cash_row(r)]
    cash_by_cat = {r["category"]: r for r in cash_rows if r["date"] == target_date}

    fii = cash_by_cat.get("FII", {})
    dii = cash_by_cat.get("DII", {})

    deriv = next(
        (d for d in deriv_data if _parse_date(d.get("date")) == target_date),
        {},
    )

    fut_long = deriv.get("fiiFutIdxLongContracts") or 0
    fut_short = deriv.get("fiiFutIdxShortContracts") or 0
    ratio = (fut_long / fut_short) if fut_short else None

    return {
        "trade_date": target_date,
        "fii_cash_buy": fii.get("buy"),
        "fii_cash_sell": fii.get("sell"),
        "fii_cash_net": fii.get("net"),
        "dii_cash_buy": dii.get("buy"),
        "dii_cash_sell": dii.get("sell"),
        "dii_cash_net": dii.get("net"),
        "fii_index_fut_long_contracts": fut_long,
        "fii_index_fut_short_contracts": fut_short,
        "fii_index_fut_net_contracts": fut_long - fut_short,
        "fii_index_fut_long_short_ratio": ratio,
        "fii_index_call_long": deriv.get("fiiOptIdxCallLongContracts") or 0,
        "fii_index_call_short": deriv.get("fiiOptIdxCallShortContracts") or 0,
        "fii_index_put_long": deriv.get("fiiOptIdxPutLongContracts") or 0,
        "fii_index_put_short": deriv.get("fiiOptIdxPutShortContracts") or 0,
        "fii_stock_fut_long": deriv.get("fiiFutStkLongContracts") or 0,
        "fii_stock_fut_short": deriv.get("fiiFutStkShortContracts") or 0,
        "raw_payload": {"cash": cash_rows, "deriv": deriv},
    }
