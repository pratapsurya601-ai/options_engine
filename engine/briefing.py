"""
Morning Briefing — one-page pre-market summary.

Combines price structure, option chain analytics, OI flow signals, and engine
rule states into a single dict consumable by /api/briefing and /briefing.

Designed to be tolerant of missing data: if a section fails (no Kite token, no
historical bars, no snapshots), the section is returned empty/partial rather
than failing the whole briefing.

CLI:
    python -m engine.briefing
"""
from __future__ import annotations

import json
import statistics
import traceback
from datetime import datetime, timedelta, time as dtime, timezone
from typing import Any


IST = timezone(timedelta(hours=5, minutes=30))


# ---------- helpers ----------

def _market_status() -> str:
    now = datetime.now(tz=IST)
    if now.weekday() >= 5:
        return "closed (weekend)"
    t = now.time()
    if dtime(9, 15) <= t <= dtime(15, 30):
        return "open"
    if t < dtime(9, 15):
        mins = (9 * 60 + 15) - (t.hour * 60 + t.minute)
        return f"pre-open (opens in {mins} min)"
    return "closed"


def _safe(fn, default=None):
    """Run fn(), swallowing any exception and returning default."""
    try:
        return fn()
    except Exception:
        return default


# ---------- section builders ----------

def _section_price(symbol: str) -> dict:
    """prev_close/high/low, today's open, current spot, gap."""
    out: dict[str, Any] = {
        "prev_close": None, "prev_high": None, "prev_low": None,
        "today_open": None, "current_spot": None,
        "gap_pts": None, "gap_pct": None,
    }
    # Daily bars
    try:
        from .data.cache import get_bars
        bars = get_bars(symbol, "day", 5)
        if bars:
            # Determine "today" vs "previous"
            today = datetime.now(tz=IST).date()
            todays = [b for b in bars if b.ts.astimezone(IST).date() == today]
            prior = [b for b in bars if b.ts.astimezone(IST).date() < today]
            if prior:
                pb = prior[-1]
                out["prev_close"] = round(pb.close, 2)
                out["prev_high"] = round(pb.high, 2)
                out["prev_low"] = round(pb.low, 2)
            if todays:
                out["today_open"] = round(todays[-1].open, 2)
    except Exception:
        pass

    # Current spot via Kite
    try:
        from .data.kite_source import spot_ltp
        out["current_spot"] = round(spot_ltp(symbol), 2)
    except Exception:
        pass

    # Gap
    if out["prev_close"] and out["today_open"]:
        gap = out["today_open"] - out["prev_close"]
        out["gap_pts"] = round(gap, 2)
        out["gap_pct"] = round(gap / out["prev_close"] * 100, 3)
    elif out["prev_close"] and out["current_spot"] and out["today_open"] is None:
        # Pre-market: use current spot as the implicit gap reference
        gap = out["current_spot"] - out["prev_close"]
        out["gap_pts"] = round(gap, 2)
        out["gap_pct"] = round(gap / out["prev_close"] * 100, 3)
    return out


def _section_daily_emas(symbol: str) -> dict:
    out: dict[str, Any] = {"ema20": None, "ema50": None, "alignment": "flat"}
    try:
        from .data.cache import get_bars
        from .indicators import ema
        bars = get_bars(symbol, "day", 120)
        if not bars or len(bars) < 50:
            return out
        closes = [b.close for b in bars]
        e20 = ema(closes, 20)
        e50 = ema(closes, 50)
        v20 = next((x for x in reversed(e20) if x is not None), None)
        v50 = next((x for x in reversed(e50) if x is not None), None)
        if v20 is not None:
            out["ema20"] = round(v20, 2)
        if v50 is not None:
            out["ema50"] = round(v50, 2)
        if v20 is not None and v50 is not None:
            spread_pct = (v20 - v50) / v50 * 100
            if spread_pct > 0.3:
                out["alignment"] = "bullish"
            elif spread_pct < -0.3:
                out["alignment"] = "bearish"
            else:
                out["alignment"] = "flat"
    except Exception:
        pass
    return out


def _section_option_chain(symbol: str, chain) -> dict:
    """ATM strike/CE/PE/IV and expiry info."""
    out: dict[str, Any] = {
        "nearest_expiry": None, "dte_days": None,
        "atm_strike": None, "atm_ce_ltp": None,
        "atm_pe_ltp": None, "atm_iv": None,
    }
    if chain is None:
        return out
    try:
        out["nearest_expiry"] = chain.expiry.isoformat()
        from .chain import t_years_to_expiry
        out["dte_days"] = round(t_years_to_expiry(chain.expiry) * 365, 2)
        spot = chain.spot
        by_strike = chain.by_strike()
        # Find ATM strike — nearest to spot
        strikes = sorted(by_strike.keys())
        if strikes:
            atm = min(strikes, key=lambda k: abs(k - spot))
            out["atm_strike"] = atm
            legs = by_strike.get(atm, {})
            ce = legs.get("CE")
            pe = legs.get("PE")
            if ce:
                out["atm_ce_ltp"] = round(ce.ltp, 2)
            if pe:
                out["atm_pe_ltp"] = round(pe.ltp, 2)
            # ATM IV = mean of CE/PE iv (where populated)
            ivs = [q.iv for q in (ce, pe) if q and q.iv]
            if ivs:
                out["atm_iv"] = round(sum(ivs) / len(ivs), 4)
            else:
                # Fallback: avg IV across strikes within 2% of spot
                near = [q.iv for q in chain.quotes
                        if q.iv and abs(q.strike - spot) / spot <= 0.02]
                if near:
                    out["atm_iv"] = round(sum(near) / len(near), 4)
    except Exception:
        pass
    return out


def _section_oi_zones(chain) -> dict:
    out: dict[str, Any] = {
        "call_wall": None, "call_wall_oi": None,
        "put_wall": None, "put_wall_oi": None,
        "max_pain": None, "zone_width": None,
    }
    if chain is None:
        return out
    try:
        from .oi_zones import compute_zones
        z = compute_zones(chain)
        out.update({
            "call_wall": z.call_wall,
            "call_wall_oi": z.call_wall_oi,
            "put_wall": z.put_wall,
            "put_wall_oi": z.put_wall_oi,
            "max_pain": z.max_pain,
            "zone_width": round(z.zone_width, 2) if z.zone_width else None,
        })
    except Exception:
        pass
    return out


def _section_expected_move(chain, atm_iv: float | None) -> dict:
    out: dict[str, Any] = {
        "one_sigma_pts": None, "range_low": None, "range_high": None,
    }
    if chain is None or atm_iv is None:
        return out
    try:
        from .probability import expected_move
        from .chain import t_years_to_expiry
        t = t_years_to_expiry(chain.expiry)
        em = expected_move(chain.spot, t, atm_iv)
        out["one_sigma_pts"] = round(em, 2)
        out["range_low"] = round(chain.spot - em, 2)
        out["range_high"] = round(chain.spot + em, 2)
    except Exception:
        pass
    return out


def _section_coi_signals(symbol: str, chain) -> list[dict]:
    """
    Compare current chain OI vs yesterday's last snapshot. Returns top 5 most
    significant changes with classified signal:
      - put_writing  (PE OI up):   bullish — writers selling puts
      - put_unwinding (PE OI down): bearish-relief or close
      - call_writing (CE OI up):    bearish — writers selling calls
      - call_unwinding (CE OI down): bullish-relief or close
    """
    if chain is None:
        return []
    try:
        from .data.chain_snapshot import read_nearest
        # Try previous trading day at 15:25
        now = datetime.now(tz=IST)
        # Walk back up to 7 days to find a snapshot
        df = None
        for back in range(1, 8):
            target = (now - timedelta(days=back)).replace(
                hour=15, minute=25, second=0, microsecond=0
            )
            df = read_nearest(symbol, target, max_delta_min=240)
            if df is not None and not df.empty:
                break
        if df is None or df.empty:
            return []

        # Build {(strike, type): oi} from snapshot
        # Filter to same expiry
        snap_expiry = chain.expiry
        df_e = df[df["expiry"] == snap_expiry] if "expiry" in df.columns else df
        prev_oi: dict[tuple[int, str], int] = {}
        for _, r in df_e.iterrows():
            try:
                k = int(r["strike"])
                t = str(r["option_type"])
                oi = r.get("oi")
                if oi and oi > 0:
                    prev_oi[(k, t)] = int(oi)
            except Exception:
                continue
        if not prev_oi:
            return []

        # Compare to current chain
        spot = chain.spot
        changes = []
        for q in chain.quotes:
            if not q.oi or q.oi <= 0:
                continue
            # Focus near spot — within ~5% to avoid noise from far strikes
            if abs(q.strike - spot) / spot > 0.05:
                continue
            prev = prev_oi.get((q.strike, q.option_type))
            if not prev or prev <= 0:
                continue
            delta = q.oi - prev
            pct = (delta / prev) * 100
            if abs(pct) < 5:
                continue
            if q.option_type == "PE":
                signal = "put_writing" if delta > 0 else "put_unwinding"
            else:
                signal = "call_writing" if delta > 0 else "call_unwinding"
            changes.append({
                "strike": q.strike,
                "type": q.option_type,
                "oi_change_pct": round(pct, 2),
                "oi_now": int(q.oi),
                "oi_prev": int(prev),
                "signal": signal,
            })
        changes.sort(key=lambda x: abs(x["oi_change_pct"]), reverse=True)
        return changes[:5]
    except Exception:
        return []


def _section_volume(symbol: str) -> dict:
    out: dict[str, Any] = {"today_vol": None, "avg_20d_vol": None, "ratio": None}
    try:
        from .data.cache import get_bars
        bars = get_bars(symbol, "day", 40)
        if not bars:
            return out
        today = datetime.now(tz=IST).date()
        todays = [b for b in bars if b.ts.astimezone(IST).date() == today]
        prior = [b for b in bars if b.ts.astimezone(IST).date() < today]
        if todays:
            out["today_vol"] = int(todays[-1].volume)
        vols = [b.volume for b in prior[-20:]]
        if vols:
            avg = statistics.median(vols)
            out["avg_20d_vol"] = int(avg)
            if avg > 0 and out["today_vol"] is not None:
                out["ratio"] = round(out["today_vol"] / avg, 2)
    except Exception:
        pass
    return out


def _section_bias(price: dict, emas: dict, coi: list[dict],
                  oi_zones: dict) -> dict:
    """
    Compose a bias score in [-1, +1] from:
      - gap (today_open vs prev_close): +/- up to 0.25
      - EMA alignment: +/- 0.30
      - COI net signal: +/- 0.25
      - max_pain vs current_spot: +/- 0.20
    """
    reasons: list[str] = []
    score = 0.0

    # Gap
    if price.get("gap_pct") is not None:
        gp = price["gap_pct"]
        if gp > 0.3:
            score += 0.25
            reasons.append(f"Gap up {gp:.2f}%")
        elif gp < -0.3:
            score -= 0.25
            reasons.append(f"Gap down {gp:.2f}%")
        else:
            reasons.append(f"Flat open ({gp:+.2f}%)")

    # EMA alignment
    align = emas.get("alignment")
    if align == "bullish":
        score += 0.30
        reasons.append("Daily EMA20 > EMA50 (uptrend)")
    elif align == "bearish":
        score -= 0.30
        reasons.append("Daily EMA20 < EMA50 (downtrend)")
    else:
        reasons.append("Daily EMAs flat")

    # COI net
    if coi:
        bull = sum(1 for c in coi if c["signal"] in ("put_writing", "call_unwinding"))
        bear = sum(1 for c in coi if c["signal"] in ("call_writing", "put_unwinding"))
        if bull > bear:
            score += 0.25
            reasons.append(f"COI bullish ({bull} bullish vs {bear} bearish signals)")
        elif bear > bull:
            score -= 0.25
            reasons.append(f"COI bearish ({bear} bearish vs {bull} bullish signals)")
        else:
            reasons.append("COI mixed")

    # Max pain vs spot
    spot = price.get("current_spot")
    mp = oi_zones.get("max_pain")
    if spot and mp:
        diff_pct = (mp - spot) / spot * 100
        if diff_pct > 0.3:
            score += 0.20
            reasons.append(f"Max pain {mp} above spot ({diff_pct:+.2f}%)")
        elif diff_pct < -0.3:
            score -= 0.20
            reasons.append(f"Max pain {mp} below spot ({diff_pct:+.2f}%)")
        else:
            reasons.append(f"Spot near max pain ({mp})")

    score = max(-1.0, min(1.0, score))
    if score >= 0.2:
        direction = "bullish"
    elif score <= -0.2:
        direction = "bearish"
    else:
        direction = "neutral"
    return {"direction": direction, "score": round(score, 2), "reasons": reasons}


def _section_summary(meta: dict, price: dict, oi_zones: dict,
                     em: dict, bias: dict, coi: list[dict]) -> str:
    parts: list[str] = []
    symbol = meta.get("symbol", "NIFTY")
    spot = price.get("current_spot")
    prev_close = price.get("prev_close")
    if spot:
        s = f"{symbol} at {spot}"
        if prev_close and price.get("gap_pct") is not None:
            s += f" ({price['gap_pct']:+.2f}% vs prev close {prev_close})"
        parts.append(s + ".")

    pw, cw = oi_zones.get("put_wall"), oi_zones.get("call_wall")
    if pw and cw:
        parts.append(f"OI zone {pw}-{cw} (width {oi_zones.get('zone_width')}); "
                     f"max pain {oi_zones.get('max_pain')}.")

    if em.get("range_low") and em.get("range_high"):
        parts.append(f"1-sigma expected range {em['range_low']}-{em['range_high']} "
                     f"(+/-{em.get('one_sigma_pts')} pts).")

    if coi:
        top = coi[0]
        parts.append(f"Top OI move: {top['signal']} at {top['strike']}{top['type']} "
                     f"({top['oi_change_pct']:+.1f}%).")

    parts.append(f"Bias {bias['direction']} (score {bias['score']:+.2f}).")
    return " ".join(parts)


# ---------- public entry ----------

def build_briefing(symbol: str = "NIFTY") -> dict:
    """
    Build a structured pre-market briefing for `symbol`.

    Tolerant of failures: each section is wrapped in try/except so a missing
    Kite token, missing snapshots, or empty cache will not break the briefing.
    """
    errors: list[str] = []

    now = datetime.now(tz=IST)
    meta = {
        "symbol": symbol,
        "generated_at_ist": now.isoformat(),
        "market_status": _market_status(),
    }

    # Pull chain once (with OI) — shared across multiple sections
    chain = None
    try:
        from .data.kite_source import option_chain, populate_ivs
        chain = option_chain(symbol, with_oi=True)
        try:
            populate_ivs(chain)
        except Exception as e:
            errors.append(f"populate_ivs: {e}")
    except Exception as e:
        errors.append(f"option_chain: {e}")

    price = _safe(lambda: _section_price(symbol), {}) or {}
    daily_emas = _safe(lambda: _section_daily_emas(symbol), {}) or {}
    option_chain_section = _safe(lambda: _section_option_chain(symbol, chain), {}) or {}
    oi_zones = _safe(lambda: _section_oi_zones(chain), {}) or {}
    em = _safe(
        lambda: _section_expected_move(chain, option_chain_section.get("atm_iv")),
        {},
    ) or {}
    coi = _safe(lambda: _section_coi_signals(symbol, chain), []) or []
    volume = _safe(lambda: _section_volume(symbol), {}) or {}
    bias = _safe(lambda: _section_bias(price, daily_emas, coi, oi_zones), {}) or {}
    summary = _safe(
        lambda: _section_summary(meta, price, oi_zones, em, bias, coi),
        "",
    ) or ""

    return {
        "meta": meta,
        "price": price,
        "daily_emas": daily_emas,
        "option_chain": option_chain_section,
        "oi_zones": oi_zones,
        "expected_move": em,
        "coi_signals": coi,
        "volume": volume,
        "bias": bias,
        "summary": summary,
        "errors": errors,
    }


if __name__ == "__main__":
    try:
        b = build_briefing("NIFTY")
        print(json.dumps(b, indent=2, default=str))
    except Exception:
        traceback.print_exc()
        raise
