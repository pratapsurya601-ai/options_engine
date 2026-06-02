"""
Flask dashboard for the options engine.

Routes:
  GET  /              dashboard HTML
  GET  /api/state     current market snapshot (spot, IV, event, IV-rank, etc.)
  GET  /api/signals   today's signals from logs/signals.jsonl
  GET  /api/positions paper positions: open + closed_today + lifetime stats
  POST /api/recommend run recommender with form params

Run:
  python -m engine.web.app
  -> open http://127.0.0.1:5050

Data source priority:
  1. Live Kite (if KITE_API_KEY set + token valid)
  2. Manual fallback — user provides spot+IV via form
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    raise SystemExit("Flask not installed. Run: pip install flask")


IST = timezone(timedelta(hours=5, minutes=30))
SIGNAL_LOG = Path("logs/signals.jsonl")
PAPER_LOG = Path("logs/paper.jsonl")


app = Flask(__name__, template_folder="templates", static_folder="static")
# Auto-reload templates on edit (no Flask restart needed for HTML/JS changes)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Research dashboard routes (/research, /api/research/*)
from .research_views import register as _register_research
_register_research(app)


# Time-based chain cache keyed by expiry — bounds Kite calls.
# Default = "nearest" expiry; positions may need other expiries (e.g. June 9).
_chain_cache: dict = {}    # key = str(expiry) or "default"
_CHAIN_TTL_SEC = 4


def _get_cached_chain(expiry=None):
    """Return chain for the given expiry. None = nearest. Caches per-expiry."""
    key = str(expiry) if expiry else "default"
    entry = _chain_cache.get(key)
    now = datetime.now(tz=IST)
    if entry and (now - entry["ts"]).total_seconds() < _CHAIN_TTL_SEC:
        return entry["chain"]
    try:
        from ..data.kite_source import option_chain, populate_ivs
        from datetime import date as _date
        # Parse expiry string if provided
        exp = None
        if expiry:
            if isinstance(expiry, str):
                exp = _date.fromisoformat(expiry)
            else:
                exp = expiry
        chain = option_chain("NIFTY", expiry=exp)
        populate_ivs(chain)
        _chain_cache[key] = {"chain": chain, "ts": now}
        return chain
    except Exception:
        return entry["chain"] if entry else None


# ------------- Helpers -------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _today_ist() -> date:
    return datetime.now(tz=IST).date()


def _live_state() -> dict:
    """Pull live snapshot via Kite. Returns None values if unavailable."""
    try:
        chain = _get_cached_chain()
        if chain is None:
            return {"source": "kite", "ok": False, "error": "no chain available"}
        spot = chain.spot
        ivs = [q.iv for q in chain.quotes
               if q.iv and abs(q.strike - spot) / spot <= 0.02]
        atm_iv = sum(ivs) / len(ivs) if ivs else None
        from ..chain import t_years_to_expiry
        t_days = t_years_to_expiry(chain.expiry) * 365
        return {
            "source": "kite",
            "ok": True,
            "spot": round(spot, 2),
            "atm_iv": round(atm_iv, 4) if atm_iv else None,
            "expiry": chain.expiry.isoformat(),
            "dte_days": round(t_days, 2),
            "snapshot_ts": chain.snapshot_ts.isoformat(),
            "quote_count": len(chain.quotes),
        }
    except Exception as e:
        return {"source": "kite", "ok": False, "error": str(e)}


def _market_status() -> str:
    now = datetime.now(tz=IST)
    if now.weekday() >= 5:
        return "closed (weekend)"
    t = now.time()
    from datetime import time as dtime
    if dtime(9, 15) <= t <= dtime(15, 30):
        return "open"
    if t < dtime(9, 15):
        return f"pre-open (opens in {((dtime(9, 15).hour*60+dtime(9, 15).minute) - (t.hour*60+t.minute))} min)"
    return "closed"


def _event_assessment(dte_days: int | None) -> dict | None:
    try:
        from ..events import assess
        if dte_days is None:
            return None
        from datetime import date, timedelta
        today = _today_ist()
        expiry = today + timedelta(days=int(dte_days))
        er = assess(today, expiry, iv_rank=None)
        return {
            "spans_event": er.spans_event,
            "events": [{"name": e.name, "date": e.event_date.isoformat()}
                       for e in er.events],
            "pre_event_now": [{"name": e.name, "date": e.event_date.isoformat()}
                              for e in er.pre_event_now],
            "advice": er.advice,
        }
    except Exception:
        return None


# ------------- Routes -------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/kite/login")
def kite_login_redirect():
    """Helper: prints the Kite login URL and offers a button to go there."""
    try:
        from ..data.kite_source import login_url
        url = login_url()
    except Exception as e:
        return f"<h1>Kite Login</h1><p>Error: {e}</p>", 500
    return f"""<!DOCTYPE html>
<html><head><title>Kite Login</title>
<style>body{{font-family:-apple-system,sans-serif;background:#0f1115;color:#e7e9ee;padding:40px;text-align:center}}
a.btn{{display:inline-block;padding:12px 24px;background:#4fc3f7;color:#061419;text-decoration:none;border-radius:6px;font-weight:600;margin:16px}}
.url{{background:#181b22;padding:12px;border-radius:6px;font-family:monospace;font-size:11px;word-break:break-all;color:#9aa3b2}}
</style></head><body>
<h1>Kite Daily Login</h1>
<p>Token expires at 6 AM IST every day. Click below to login.</p>
<a class="btn" href="{url}">Login to Kite</a>
<p>After login, Kite will redirect back here and auto-save your token.</p>
<div class="url">{url}</div>
</body></html>"""


@app.route("/kite/callback")
def kite_callback():
    """
    Kite Connect redirect target. Configure your Kite app's Redirect URL to:
        http://127.0.0.1:5050/kite/callback
    After login, Kite redirects here with ?request_token=XXX&action=login&status=success.
    We exchange the token, save it, and redirect to dashboard — no copy-paste needed.
    """
    req_token = request.args.get("request_token")
    status = request.args.get("status")
    if not req_token:
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;background:#0f1115;color:#e7e9ee;padding:40px;text-align:center">
<h1>Missing request_token</h1>
<p>Status: {status or 'unknown'}</p>
<p><a href="/kite/login" style="color:#4fc3f7">Try login again</a></p>
</body></html>""", 400
    try:
        from ..data.kite_source import login_with_request_token
        access_token = login_with_request_token(req_token)
        # Invalidate cached chain so next state-pull uses the new auth
        _chain_cache.clear()
        return f"""<!DOCTYPE html><html><head>
<meta http-equiv="refresh" content="3; url=/">
<title>Kite Login Success</title>
<style>body{{font-family:-apple-system,sans-serif;background:#0f1115;color:#e7e9ee;padding:40px;text-align:center}}
.ok{{color:#66bb6a;font-size:48px;margin:0}}
a{{color:#4fc3f7;text-decoration:none}}</style>
</head><body>
<p class="ok">✓</p>
<h1>Logged in to Kite</h1>
<p>Access token saved (len={len(access_token)}). Redirecting to dashboard in 3 seconds…</p>
<p><a href="/">Go to dashboard now</a></p>
</body></html>"""
    except Exception as e:
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;background:#0f1115;color:#e7e9ee;padding:40px;text-align:center">
<h1>Kite login failed</h1>
<pre style="color:#ef5350">{e}</pre>
<p><a href="/kite/login" style="color:#4fc3f7">Try again</a> — make sure your redirect URL in <a href="https://developers.kite.trade/apps" style="color:#4fc3f7">developers.kite.trade/apps</a> is set to <code>http://127.0.0.1:5050/kite/callback</code></p>
</body></html>""", 400


@app.route("/api/state")
def api_state():
    state = _live_state()
    state["market_status"] = _market_status()
    state["ts"] = datetime.now(tz=IST).isoformat()
    if state.get("ok") and state.get("dte_days"):
        state["event_risk"] = _event_assessment(state["dte_days"])
    return jsonify(state)


@app.route("/api/signals")
def api_signals():
    all_signals = _read_jsonl(SIGNAL_LOG)
    today = _today_ist().isoformat()
    today_sigs = [s for s in all_signals if s.get("ts", "").startswith(today)]
    return jsonify({
        "today": today_sigs,
        "total_lifetime": len(all_signals),
    })


@app.route("/api/positions")
def api_positions():
    """Reconstruct positions from paper.jsonl OPEN/CLOSE events."""
    events = _read_jsonl(PAPER_LOG)
    positions: dict[str, dict] = {}
    for ev in events:
        if ev.get("kind") == "OPEN":
            # Use the SAME pid format the watcher writes: jsonl:{entry_ts}:{strike}{ot}
            pid = f"jsonl:{ev.get('entry_ts')}:{ev.get('strike')}{ev.get('option_type')}"
            positions[pid] = {
                **ev, "status": "open",
                "id": pid,
            }
        elif ev.get("kind") == "CLOSE":
            pid = ev.get("position_id")
            if pid and pid in positions:
                # Only update if not already closed (idempotency)
                if positions[pid].get("status") != "closed":
                    positions[pid].update({
                        "exit_price": ev.get("exit_price"),
                        "exit_ts": ev.get("exit_ts"),
                        "exit_reason": ev.get("exit_reason"),
                        "pnl": ev.get("pnl"),
                        "status": "closed",
                    })
            # Removed sloppy fallback — if pid doesn't match, the CLOSE event is orphaned

    today = _today_ist().isoformat()
    all_pos = list(positions.values())
    open_pos = [p for p in all_pos if p["status"] == "open"]
    closed_today = [p for p in all_pos
                    if p["status"] == "closed" and p.get("exit_ts", "").startswith(today)]
    all_closed = [p for p in all_pos if p["status"] == "closed"]

    # --- Enrich open positions with live LTP + P&L (using each position's own expiry) ---
    if open_pos:
        # Group positions by expiry so we pull each chain once
        by_expiry: dict = {}
        for p in open_pos:
            by_expiry.setdefault(p.get("expiry"), []).append(p)

        for expiry, group in by_expiry.items():
            chain = _get_cached_chain(expiry=expiry)
            if chain is None:
                continue
            by_strike = chain.by_strike()
            for p in group:
                strike = p.get("strike")
                ot = p.get("option_type")
                lots = p.get("lots") or 1
                lot_size = p.get("lot_size") or 75
                entry = p.get("entry_price")
                if strike and ot and entry:
                    q = by_strike.get(strike, {}).get(ot)
                    if q and q.ltp:
                        p["current_price"] = round(q.ltp, 2)
                        p["live_pnl"] = round(
                            (q.ltp - float(entry)) * lots * lot_size, 0
                        )
                        p["live_pnl_pct"] = round(
                            (q.ltp - float(entry)) / float(entry) * 100, 2
                        )
                        # Note which expiry chain was used (for debugging)
                        p["chain_expiry"] = str(chain.expiry)

    wins = sum(1 for p in all_closed if (p.get("pnl") or 0) > 0)
    losses = sum(1 for p in all_closed if (p.get("pnl") or 0) <= 0)
    total_pnl = sum(p.get("pnl") or 0 for p in all_closed)
    win_rate = wins / len(all_closed) if all_closed else None

    return jsonify({
        "open": open_pos,
        "closed_today": closed_today,
        "stats_lifetime": {
            "trades": len(all_closed),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
            "total_pnl_inr": round(total_pnl, 0),
        },
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Run the recommender with form-supplied parameters."""
    try:
        data = request.json or {}
        spot = float(data.get("spot"))
        iv = float(data.get("iv"))
        dte = int(data.get("dte", 7))
        view = data.get("view", "neutral")
        conviction = float(data.get("conviction", 0.5))
        win = float(data.get("win", 0.6))
        capital = float(data.get("capital", 50000))
        scalp_mode = bool(data.get("scalp", False))
        target_pts = float(data.get("target_pts", 15))
        hold_days = float(data.get("hold_days", 1.0))

        from ..manual_chain import synthetic_chain
        from ..recommender import recommend
        from ..events import assess as assess_events
        chain = synthetic_chain(symbol="NIFTY", spot=spot, iv=iv, dte_days=dte)
        t_years = dte / 365.0

        if scalp_mode:
            from ..scalp import scalp_recommend
            ideas = scalp_recommend(
                chain=chain, t_years_to_expiry=t_years,
                view=view, conviction=conviction,
                target_points=target_pts, hold_days=hold_days,
                max_premium_per_lot=min(capital, 20000),
                capital_inr=capital, top_n=5,
            )
            return jsonify({
                "ok": True, "mode": "scalp",
                "spot": spot, "iv": iv, "dte": dte,
                "ideas": [i.summary() for i in ideas],
            })

        # Standard expiry-based recommender
        today = _today_ist()
        from datetime import timedelta
        expiry = today + timedelta(days=dte)
        event_risk = assess_events(today, expiry, iv_rank=None)

        result = recommend(
            chain=chain, t_years=t_years,
            view=view, conviction=conviction,
            target_win_pct=win, max_capital_inr=capital,
            smart_money_score=0.0, top_n=5,
            event_risk=event_risk, iv_rank_info=None,
            include_rejected=True,
        )
        qualified, rejected = result
        return jsonify({
            "ok": True, "mode": "strategy",
            "spot": spot, "iv": iv, "dte": dte,
            "event_risk": {
                "spans_event": event_risk.spans_event,
                "events": [e.name + " on " + e.event_date.isoformat()
                           for e in event_risk.events],
                "advice": event_risk.advice,
            },
            "qualified": [i.summary() for i in qualified],
            "rejected": [i.summary() for i in rejected[:3]],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}), 400


@app.route("/api/briefing")
def api_briefing():
    from ..briefing import build_briefing
    return jsonify(build_briefing("NIFTY"))


@app.route("/api/paper_pnl_curve")
def api_paper_pnl_curve():
    """Cumulative paper P&L from closed events in paper.jsonl (last 30 days)."""
    events = _read_jsonl(PAPER_LOG)
    closes: list[dict] = []
    cutoff = datetime.now(tz=IST) - timedelta(days=30)
    cum = 0.0
    for ev in events:
        if ev.get("kind") != "CLOSE":
            continue
        ts = ev.get("exit_ts") or ev.get("ts")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=IST)
        if t < cutoff:
            continue
        pnl = ev.get("pnl") or ev.get("realized_pnl") or 0
        try:
            cum += float(pnl)
        except (TypeError, ValueError):
            continue
        closes.append({"ts": ts, "cum_pnl": round(cum, 2),
                       "trade_pnl": round(float(pnl), 2)})
    closes.sort(key=lambda d: d["ts"])
    return jsonify({"points": closes, "final_cum": round(cum, 2)})


@app.route("/api/rule_fires")
def api_rule_fires():
    """Counts of signal fires grouped by rule_name (last 7 days)."""
    sigs = _read_jsonl(SIGNAL_LOG)
    cutoff = (datetime.now(tz=IST) - timedelta(days=7)).isoformat()
    counts: dict[str, int] = {}
    for s in sigs:
        ts = s.get("ts", "")
        if ts < cutoff:
            continue
        rule = s.get("rule_name") or s.get("rule") or "unknown"
        counts[rule] = counts.get(rule, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return jsonify({"rules": [k for k, _ in items],
                    "counts": [v for _, v in items]})


@app.route("/api/briefing_chart")
def api_briefing_chart():
    """OI profile (per strike CE/PE OI) + bias components for charts."""
    out = {"strikes": [], "ce_oi": [], "pe_oi": [],
           "spot": None, "max_pain": None,
           "bias_components": []}
    try:
        chain = _get_cached_chain()
        if chain is not None:
            out["spot"] = round(chain.spot, 2)
            by_strike = chain.by_strike()
            strikes = sorted(by_strike.keys())
            # narrow to within 5% of spot for readability
            band = [k for k in strikes if abs(k - chain.spot) / chain.spot <= 0.05]
            for k in band:
                ce = by_strike[k].get("CE")
                pe = by_strike[k].get("PE")
                out["strikes"].append(k)
                out["ce_oi"].append(int(getattr(ce, "oi", 0) or 0) if ce else 0)
                out["pe_oi"].append(int(getattr(pe, "oi", 0) or 0) if pe else 0)
    except Exception as e:
        out["error"] = str(e)

    # Bias components from briefing
    try:
        from ..briefing import build_briefing
        b = build_briefing("NIFTY")
        out["max_pain"] = (b.get("oi_zones") or {}).get("max_pain")
        bias = b.get("bias") or {}
        # Heuristic component extraction from reasons (best effort).
        # Real components live inside bias.reasons (strings); we expose a simple
        # score mapping so the chart can render *something* honest.
        score = bias.get("score") or 0
        out["bias_components"] = [
            {"label": "Net Bias", "value": round(float(score), 3)},
        ]
        # Add structured components if present
        for k in ("gap_component", "ema_component",
                  "coi_component", "max_pain_component"):
            if k in bias:
                out["bias_components"].append(
                    {"label": k.replace("_component", "").replace("_", " ").title(),
                     "value": round(float(bias[k] or 0), 3)})
    except Exception as e:
        out.setdefault("error", str(e))
    return jsonify(out)


@app.route("/briefing")
def briefing_page():
    from ..briefing import build_briefing
    b = build_briefing("NIFTY")

    def fmt(v, suffix=""):
        if v is None:
            return "<span class='na'>—</span>"
        return f"{v}{suffix}"

    meta = b.get("meta", {})
    price = b.get("price", {})
    emas = b.get("daily_emas", {})
    oc = b.get("option_chain", {})
    oz = b.get("oi_zones", {})
    em = b.get("expected_move", {})
    coi = b.get("coi_signals", []) or []
    vol = b.get("volume", {})
    bias = b.get("bias", {})
    summary = b.get("summary", "")
    errors = b.get("errors", []) or []

    spot = price.get("current_spot")
    gap_pct = price.get("gap_pct")
    gap_pts = price.get("gap_pts")
    gap_class = "neutral"
    gap_str = "—"
    if gap_pct is not None:
        gap_class = "up" if gap_pct >= 0 else "down"
        sign = "+" if gap_pts is not None and gap_pts >= 0 else ""
        gap_str = f"{sign}{gap_pts} pts ({sign}{gap_pct}%)"

    bias_dir = bias.get("direction", "neutral")
    bias_score = bias.get("score", 0)
    bias_reasons = bias.get("reasons", []) or []
    bias_class = {"bullish": "up", "bearish": "down"}.get(bias_dir, "neutral")

    coi_rows = ""
    if coi:
        for c in coi:
            sig = c.get("signal", "")
            sig_class = "up" if sig in ("put_writing", "call_unwinding") else "down"
            coi_rows += (
                f"<tr><td>{c.get('strike')}</td><td>{c.get('type')}</td>"
                f"<td class='{sig_class}'>{c.get('oi_change_pct')}%</td>"
                f"<td class='{sig_class}'>{sig}</td></tr>"
            )
    else:
        coi_rows = "<tr><td colspan='4' class='na'>No snapshot from prior session available</td></tr>"

    reasons_html = "".join(f"<li>{r}</li>" for r in bias_reasons) or "<li class='na'>No signals</li>"
    errors_html = ""
    if errors:
        errors_html = "<div class='panel errors'><h3>Notes</h3><ul>" + \
            "".join(f"<li>{e}</li>" for e in errors) + "</ul></div>"

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Morning Briefing — {meta.get('symbol','NIFTY')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1115; color: #e7e9ee; margin: 0; padding: 16px; }}
a {{ color: #4fc3f7; text-decoration: none; }}
.top {{ display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; margin-bottom: 16px; gap: 12px; }}
.ticker {{ font-size: 28px; font-weight: 700; }}
.ticker .sym {{ color: #9aa3b2; font-size: 18px; margin-right: 8px; }}
.gap {{ font-size: 14px; margin-left: 12px; }}
.up {{ color: #66bb6a; }}
.down {{ color: #ef5350; }}
.neutral {{ color: #9aa3b2; }}
.na {{ color: #5a6271; font-style: italic; }}
.meta-line {{ color: #9aa3b2; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
         gap: 12px; }}
.panel {{ background: #181b22; border-radius: 8px; padding: 14px;
          border: 1px solid #232730; }}
.panel h3 {{ margin: 0 0 10px 0; font-size: 13px; text-transform: uppercase;
             letter-spacing: 0.5px; color: #9aa3b2; }}
.kv {{ display: flex; justify-content: space-between; padding: 4px 0;
       border-bottom: 1px dashed #232730; font-size: 14px; }}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{ color: #9aa3b2; }}
.kv .v {{ font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #232730; }}
th {{ color: #9aa3b2; font-weight: 500; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px; }}
.summary {{ font-size: 15px; line-height: 1.5; }}
.bias-badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px;
               font-weight: 600; font-size: 12px; text-transform: uppercase; }}
.bias-badge.up {{ background: #143d24; color: #66bb6a; }}
.bias-badge.down {{ background: #3d1414; color: #ef5350; }}
.bias-badge.neutral {{ background: #232730; color: #9aa3b2; }}
.reasons {{ margin: 8px 0 0 16px; padding: 0; font-size: 13px; }}
.reasons li {{ margin: 2px 0; color: #c5cad4; }}
.errors {{ background: #2a1c1c; border-color: #4d2a2a; }}
.errors li {{ color: #f5a5a5; font-size: 12px; font-family: monospace; }}
@media (max-width: 600px) {{
    .ticker {{ font-size: 22px; }}
    body {{ padding: 10px; }}
}}
/* polish */
body {{ font-feature-settings: "tnum" 1, "lnum" 1; line-height: 1.45; }}
h1, h2, h3 {{ line-height: 1.2; }}
.panel {{ transition: transform 120ms ease, border-color 120ms ease; }}
.panel:hover {{ transform: translateY(-1px); border-color: #2c313c; }}
table thead th {{ position: sticky; top: 0; background: #181b22; z-index: 1; }}
.bias-badge {{ font-size: 11px; }}
.chart-wrap {{ position: relative; height: 220px; }}
.chart-wrap.tall {{ height: 280px; }}
.empty-state {{ color: #6b7280; font-size: 12px; text-align: center;
                padding: 24px 12px; font-style: italic; }}
</style></head>
<body>

<div class="top">
  <div>
    <div class="ticker">
      <span class="sym">{meta.get('symbol','NIFTY')}</span>
      <span>{fmt(spot)}</span>
      <span class="gap {gap_class}">{gap_str}</span>
    </div>
    <div class="meta-line">
      Generated {meta.get('generated_at_ist','')} — Market: {meta.get('market_status','')}
    </div>
  </div>
  <div><a href="/">← Dashboard</a></div>
</div>

<div class="panel summary" style="margin-bottom:12px">
  <h3>Summary</h3>
  <p style="margin:0">{summary or '<span class="na">No summary available</span>'}</p>
  <div style="margin-top:10px">
    <span class="bias-badge {bias_class}">{bias_dir} {bias_score:+.2f}</span>
  </div>
  <ul class="reasons">{reasons_html}</ul>
</div>

<div class="grid">

  <div class="panel">
    <h3>Price Structure</h3>
    <div class="kv"><span class="k">Prev Close</span><span class="v">{fmt(price.get('prev_close'))}</span></div>
    <div class="kv"><span class="k">Prev High</span><span class="v">{fmt(price.get('prev_high'))}</span></div>
    <div class="kv"><span class="k">Prev Low</span><span class="v">{fmt(price.get('prev_low'))}</span></div>
    <div class="kv"><span class="k">Today Open</span><span class="v">{fmt(price.get('today_open'))}</span></div>
    <div class="kv"><span class="k">Current Spot</span><span class="v">{fmt(spot)}</span></div>
    <div class="kv"><span class="k">Gap</span><span class="v {gap_class}">{gap_str}</span></div>
    <div class="kv"><span class="k">EMA20 (D)</span><span class="v">{fmt(emas.get('ema20'))}</span></div>
    <div class="kv"><span class="k">EMA50 (D)</span><span class="v">{fmt(emas.get('ema50'))}</span></div>
    <div class="kv"><span class="k">Alignment</span><span class="v">{emas.get('alignment','—')}</span></div>
  </div>

  <div class="panel">
    <h3>Option Chain (ATM)</h3>
    <div class="kv"><span class="k">Expiry</span><span class="v">{fmt(oc.get('nearest_expiry'))}</span></div>
    <div class="kv"><span class="k">DTE</span><span class="v">{fmt(oc.get('dte_days'),' d')}</span></div>
    <div class="kv"><span class="k">ATM Strike</span><span class="v">{fmt(oc.get('atm_strike'))}</span></div>
    <div class="kv"><span class="k">ATM CE LTP</span><span class="v">{fmt(oc.get('atm_ce_ltp'))}</span></div>
    <div class="kv"><span class="k">ATM PE LTP</span><span class="v">{fmt(oc.get('atm_pe_ltp'))}</span></div>
    <div class="kv"><span class="k">ATM IV</span><span class="v">{fmt(round(oc.get('atm_iv')*100,2) if oc.get('atm_iv') else None, '%')}</span></div>
  </div>

  <div class="panel">
    <h3>OI Zones</h3>
    <div class="kv"><span class="k">Call Wall</span><span class="v down">{fmt(oz.get('call_wall'))}</span></div>
    <div class="kv"><span class="k">Call Wall OI</span><span class="v">{fmt(oz.get('call_wall_oi'))}</span></div>
    <div class="kv"><span class="k">Put Wall</span><span class="v up">{fmt(oz.get('put_wall'))}</span></div>
    <div class="kv"><span class="k">Put Wall OI</span><span class="v">{fmt(oz.get('put_wall_oi'))}</span></div>
    <div class="kv"><span class="k">Max Pain</span><span class="v">{fmt(oz.get('max_pain'))}</span></div>
    <div class="kv"><span class="k">Zone Width</span><span class="v">{fmt(oz.get('zone_width'))}</span></div>
  </div>

  <div class="panel">
    <h3>Expected Move (1σ)</h3>
    <div class="kv"><span class="k">±1σ</span><span class="v">{fmt(em.get('one_sigma_pts'),' pts')}</span></div>
    <div class="kv"><span class="k">Range Low</span><span class="v down">{fmt(em.get('range_low'))}</span></div>
    <div class="kv"><span class="k">Range High</span><span class="v up">{fmt(em.get('range_high'))}</span></div>
  </div>

  <div class="panel">
    <h3>COI Signals (vs prior session)</h3>
    <table>
      <thead><tr><th>Strike</th><th>Type</th><th>ΔOI%</th><th>Signal</th></tr></thead>
      <tbody>{coi_rows}</tbody>
    </table>
  </div>

  <div class="panel">
    <h3>Volume</h3>
    <div class="kv"><span class="k">Today Vol</span><span class="v">{fmt(vol.get('today_vol'))}</span></div>
    <div class="kv"><span class="k">Avg 20d Vol</span><span class="v">{fmt(vol.get('avg_20d_vol'))}</span></div>
    <div class="kv"><span class="k">Ratio</span><span class="v">{fmt(vol.get('ratio'),'×')}</span></div>
  </div>

  <div class="panel" style="grid-column: span 2;">
    <h3>OI Profile (CE vs PE per strike)</h3>
    <div class="chart-wrap tall"><canvas id="oiProfileChart"></canvas></div>
    <div id="oiProfileEmpty" class="empty-state" style="display:none">
      No chain data available yet. Charts populate when Kite session is live.
    </div>
  </div>

  <div class="panel">
    <h3>Bias Breakdown</h3>
    <div class="chart-wrap"><canvas id="biasChart"></canvas></div>
    <div id="biasEmpty" class="empty-state" style="display:none">
      No bias components published.
    </div>
  </div>

</div>

{errors_html}

<div style="margin-top:20px;color:#5a6271;font-size:11px;text-align:center">
  Auto-refresh every 60s · <a href="/api/briefing">JSON</a>
</div>

<script>
if (window.Chart) {{
  Chart.defaults.color = '#9aa3b2';
  Chart.defaults.borderColor = '#232730';
  Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
}}
fetch('/api/briefing_chart').then(r => r.json()).then(d => {{
  // OI Profile
  if (!d.strikes || !d.strikes.length) {{
    document.getElementById('oiProfileEmpty').style.display = 'block';
    document.getElementById('oiProfileChart').style.display = 'none';
  }} else {{
    const ctx = document.getElementById('oiProfileChart').getContext('2d');
    const spot = d.spot;
    const mp = d.max_pain;
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: d.strikes,
        datasets: [
          {{ label: 'CE OI', data: d.ce_oi, backgroundColor: 'rgba(239,83,80,0.65)' }},
          {{ label: 'PE OI', data: d.pe_oi, backgroundColor: 'rgba(102,187,106,0.65)' }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{ x: {{ stacked: false }}, y: {{ beginAtZero: true }} }},
        plugins: {{
          legend: {{ position: 'top' }},
          tooltip: {{ callbacks: {{
            afterBody: function(items) {{
              const lines = [];
              if (spot != null) lines.push('Spot: ' + spot);
              if (mp != null) lines.push('Max Pain: ' + mp);
              return lines;
            }}
          }} }}
        }}
      }}
    }});
  }}
  // Bias breakdown
  const bc = d.bias_components || [];
  if (!bc.length) {{
    document.getElementById('biasEmpty').style.display = 'block';
    document.getElementById('biasChart').style.display = 'none';
  }} else {{
    const ctx2 = document.getElementById('biasChart').getContext('2d');
    new Chart(ctx2, {{
      type: 'bar',
      data: {{
        labels: bc.map(c => c.label),
        datasets: [{{
          label: 'Bias',
          data: bc.map(c => c.value),
          backgroundColor: bc.map(c => c.value >= 0 ? 'rgba(102,187,106,0.7)' : 'rgba(239,83,80,0.7)'),
        }}]
      }},
      options: {{
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ beginAtZero: true }} }}
      }}
    }});
  }}
}}).catch(err => {{
  console.error('briefing chart load failed', err);
  document.getElementById('oiProfileEmpty').style.display = 'block';
  document.getElementById('biasEmpty').style.display = 'block';
}});
</script>

</body></html>"""
    return html


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    print(f"Options Engine dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
