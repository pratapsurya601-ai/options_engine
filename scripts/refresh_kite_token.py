"""
Auto-refresh Kite Connect access token via TOTP 2FA.

Logs in to Zerodha programmatically using stored credentials + TOTP secret,
follows the redirect chain to capture the request_token without needing the
local callback server to be running, then exchanges the request_token for an
access_token via the official KiteConnect SDK.

Designed to run unattended in a GitHub Actions cron job (or locally as a
scheduled task) every weekday before 9:15 AM IST so the chain-snapshot
workflow always has a valid token.

Env vars (all required):
  KITE_API_KEY
  KITE_API_SECRET
  ZERODHA_USER_ID       — your Zerodha login id (e.g. 'AB1234')
  ZERODHA_PASSWORD      — your account password
  ZERODHA_TOTP_SECRET   — base32 TOTP secret from External 2FA setup

Usage:
  python scripts/refresh_kite_token.py           # prints token to stdout
  python scripts/refresh_kite_token.py --json    # prints {"access_token": "..."}
  python scripts/refresh_kite_token.py --verify  # also validates the token

Exits 0 on success, non-zero on failure. Token is masked from logs in CI.

ToS note: this uses Zerodha's web login endpoints (not the documented API
flow) and is technically a gray area. Used widely by Indian quant traders
without incident, but Zerodha may change the flow or detect/block automated
login at any time. If your account has live funds, prefer the manual flow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import parse_qs, urlparse

LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"ERROR: required env var {name} is not set", file=sys.stderr)
        sys.exit(2)
    return v


def _capture_request_token(session, connect_url: str, max_hops: int = 12) -> str:
    """Follow redirect chain until we find request_token in a URL.

    The final redirect lands on YOUR registered redirect URL (e.g.
    http://127.0.0.1:5050/kite/callback?request_token=...). On a CI runner
    that URL is unreachable, but the request_token is already in the
    Location header before the final fetch — so we extract it without
    actually visiting the local callback.
    """
    import requests

    url = connect_url
    for _ in range(max_hops):
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "request_token" in qs and qs["request_token"][0]:
            return qs["request_token"][0]

        try:
            r = session.get(url, allow_redirects=False, timeout=15)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Connection error while following redirects (last URL: {url}): {e}"
            )

        if r.status_code in (301, 302, 303, 307, 308):
            nxt = r.headers.get("Location", "")
            if not nxt:
                raise RuntimeError(f"Redirect at {url} had no Location header")
            if nxt.startswith("/"):
                nxt = f"{parsed.scheme}://{parsed.netloc}{nxt}"
            url = nxt
            continue

        # Not a redirect. Check current response URL one last time.
        final_qs = parse_qs(urlparse(r.url).query)
        if "request_token" in final_qs and final_qs["request_token"][0]:
            return final_qs["request_token"][0]
        raise RuntimeError(
            f"Redirect chain ended without request_token at {r.url} "
            f"(status {r.status_code})"
        )

    raise RuntimeError(f"Too many redirects (>{max_hops}) without finding request_token")


def login_and_get_access_token() -> str:
    import pyotp
    import requests

    api_key = _require_env("KITE_API_KEY")
    api_secret = _require_env("KITE_API_SECRET")
    user_id = _require_env("ZERODHA_USER_ID")
    password = _require_env("ZERODHA_PASSWORD")
    totp_secret = _require_env("ZERODHA_TOTP_SECRET")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; options-engine-token-refresh/1.0)",
    })

    # ---- Step 1: user_id + password ----
    r = session.post(
        LOGIN_URL,
        data={"user_id": user_id, "password": password},
        timeout=15,
    )
    try:
        body = r.json()
    except ValueError:
        raise RuntimeError(
            f"Step 1 (login) returned non-JSON (status {r.status_code}): "
            f"{r.text[:200]}"
        )
    if body.get("status") != "success":
        raise RuntimeError(f"Step 1 (login) failed: {body}")
    request_id = body["data"]["request_id"]

    # ---- Step 2: 2FA via TOTP ----
    totp_code = pyotp.TOTP(totp_secret).now()
    r = session.post(
        TWOFA_URL,
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_code,
            "twofa_type": "totp",
            "skip_session": "",
        },
        timeout=15,
    )
    try:
        body = r.json()
    except ValueError:
        raise RuntimeError(
            f"Step 2 (2FA) returned non-JSON (status {r.status_code}): "
            f"{r.text[:200]}"
        )
    if body.get("status") != "success":
        raise RuntimeError(f"Step 2 (2FA) failed: {body}")

    # ---- Step 3: visit connect URL, extract request_token from redirect chain ----
    connect_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    request_token = _capture_request_token(session, connect_url)

    # ---- Step 4: exchange request_token for access_token via official SDK ----
    try:
        from kiteconnect import KiteConnect
    except ImportError as e:
        raise RuntimeError(f"kiteconnect SDK not installed: {e}")
    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data.get("access_token")
    if not access_token:
        raise RuntimeError(f"generate_session returned no access_token: {session_data}")
    return access_token


def verify_token(token: str) -> bool:
    """Call kite.profile() to confirm the token actually works before we
    overwrite the existing one."""
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
        kite.set_access_token(token)
        prof = kite.profile()
        return bool(prof.get("user_id"))
    except Exception as e:
        print(f"Token verification failed: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Write token to this file (default: stdout)")
    p.add_argument("--json", action="store_true", help="Print as JSON object")
    p.add_argument("--verify", action="store_true",
                   help="Call kite.profile() to confirm the token works before exiting")
    args = p.parse_args()

    try:
        token = login_and_get_access_token()
    except Exception as e:
        print(f"Auto-refresh FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verify and not verify_token(token):
        print("Token returned by login flow but verification call failed.",
              file=sys.stderr)
        sys.exit(3)

    payload = json.dumps({"access_token": token}) if args.json else token

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
