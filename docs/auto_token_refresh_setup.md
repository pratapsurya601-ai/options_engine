# Automated Daily Kite Token Refresh

> WARNING — Read the risk section before enabling this. You are storing your
> full Zerodha credentials + TOTP secret in GitHub Secrets. If those leak,
> an attacker can log in as you and place trades on your account (they
> CANNOT withdraw funds — Zerodha withdrawals require linked-bank UPI/OTP).

## What this does

Every weekday at 8:30 AM IST, a GitHub Actions workflow:

1. Logs in to Zerodha using your stored `user_id` + `password`
2. Computes a TOTP code from your stored `totp_secret`
3. Submits 2FA
4. Captures the new `request_token` from Kite's redirect chain
5. Exchanges it for an `access_token` via the official KiteConnect SDK
6. Verifies the token works with a `kite.profile()` call
7. Updates the `KITE_ACCESS_TOKEN` repo secret using `gh secret set`

Result: the chain_snapshot workflow at 9:15 AM IST always has a fresh token.
Zero daily manual work.

## When NOT to use this

- Your Kite API key has order permissions AND your account has substantial
  balance. (Use the manual flow instead — 2 minutes per day vs your capital.)
- You're not comfortable with the gray-area ToS implications (Zerodha does
  not officially support automated login).

## When this is reasonable

- You're paper trading only (no balance to drain).
- You've created a Kite app dedicated to data fetching that you don't use
  for live orders.
- You accept that Zerodha may change the login flow at any time and break
  this workflow until you patch it.

## Setup (one-time, ~15 minutes)

### Step 1: Enable External TOTP 2FA on Zerodha

Zerodha defaults to SMS OTP. You need to switch to External Authenticator
(Google Authenticator / Authy style) which gives you a TOTP secret.

1. Login to Kite web at https://kite.zerodha.com
2. Profile → Settings → Two-factor authentication
3. Change to **External 2FA / Authenticator App**
4. Zerodha displays a QR code and a text secret (base32 string like
   `JBSWY3DPEHPK3PXP...`)
5. Scan the QR with Google Authenticator on your phone (so you can still
   log in manually if needed)
6. **Copy the text secret** — this is your `ZERODHA_TOTP_SECRET`. Save it
   somewhere safe before the page closes. If you lose it, you have to
   re-enable 2FA from scratch.

Verify TOTP works by logging out and logging back in using a code from
your authenticator app.

### Step 2: Create a GitHub Personal Access Token (PAT)

The workflow needs to write a repo secret, which requires elevated
permissions beyond the default `GITHUB_TOKEN`.

1. Open https://github.com/settings/tokens/new
2. **Note**: `options_engine kite token refresh`
3. **Expiration**: 90 days (or whatever you're comfortable rotating)
4. **Scopes**: tick `repo` (full control of repos — needed to update
   secrets via the API)
5. Click **Generate token**
6. Copy the token immediately (`ghp_...`). You will not see it again.

### Step 3: Add the new GitHub Secrets

Open https://github.com/pratapsurya601-ai/options_engine/settings/secrets/actions
and click **New repository secret** four times:

| Name | Value |
|---|---|
| `ZERODHA_USER_ID` | Your Zerodha login id (e.g. `AB1234`) |
| `ZERODHA_PASSWORD` | Your account password |
| `ZERODHA_TOTP_SECRET` | Base32 string from Step 1 (no spaces) |
| `GH_PAT` | The `ghp_...` token from Step 2 |

You should now have 8 secrets total:

```
DATABASE_URL
KITE_API_KEY
KITE_API_SECRET
KITE_ACCESS_TOKEN
ZERODHA_USER_ID
ZERODHA_PASSWORD
ZERODHA_TOTP_SECRET
GH_PAT
```

### Step 4: Test the workflow manually

1. Go to https://github.com/pratapsurya601-ai/options_engine/actions
2. Click **"Refresh Kite Token"**
3. Click **Run workflow** → main → **Run workflow**
4. Watch the run. Two steps to expect:
   - **Refresh and verify token** → should print no token (it's masked)
     and complete in 5-15 seconds
   - **Update KITE_ACCESS_TOKEN repo secret** → should print
     `KITE_ACCESS_TOKEN updated successfully.`
5. Verify by going to Settings → Secrets and noting that the
   `Updated` timestamp on `KITE_ACCESS_TOKEN` changed.

If both steps green: you're done. From tomorrow, the cron at 8:30 AM IST
fires automatically.

## What you'll see going forward

Every weekday morning, **without you doing anything**:

```
03:00 UTC (8:30 IST)  refresh_kite_token.yml fires
                      logs in, gets new access_token, writes secret
03:45 UTC (9:15 IST)  chain_snapshot.yml fires (first market-open capture)
                      uses the freshly-rotated KITE_ACCESS_TOKEN
...                   chain_snapshot fires every 5 min until 10:00 UTC
10:00 UTC (15:30 IST) last capture of the day
```

You can sleep through it. You can travel. You can ignore your laptop. The
research dataset just grows.

## Failure modes and what to do

| Failure | Symptom | Fix |
|---|---|---|
| Wrong TOTP secret | "Step 2 (2FA) failed" | Re-do Step 1, copy the secret carefully |
| Zerodha added CAPTCHA | "Step 1 returned non-JSON" with HTML in body | Patch the script to handle the new flow, or fall back to manual |
| Account locked | "Step 1 failed: account locked" | Login manually via browser to unlock; check Zerodha email for security alerts |
| GH_PAT expired | "gh: HTTP 401 Unauthorized" | Regenerate at github.com/settings/tokens, update `GH_PAT` secret |
| GH_PAT lacks scope | "gh: HTTP 403" on secret update | Token needs the `repo` scope (not just `public_repo`) |

When the auto-refresh fails, the chain_snapshot workflow at 9:15 AM will
also fail (stale token). You'll get email notifications from GitHub for both.
Fix the cause, then run `Refresh Kite Token` manually once to recover.

## Rotation hygiene

- **`GH_PAT`** expires in 90 days. Set a calendar reminder to regenerate.
- **`ZERODHA_PASSWORD`** changes whenever you rotate it manually. Update
  the secret immediately after.
- **`ZERODHA_TOTP_SECRET`** changes only if you re-enable 2FA from scratch.
  Usually never.

## Disabling

If anything feels off (suspicious account activity, ToS concerns, etc):

1. Disable the workflow:
   ```
   gh workflow disable refresh_kite_token.yml
   ```
   Or via web: Actions → "Refresh Kite Token" → "..." → Disable workflow.
2. Delete the four sensitive secrets via the GitHub web UI:
   `ZERODHA_USER_ID`, `ZERODHA_PASSWORD`, `ZERODHA_TOTP_SECRET`, `GH_PAT`
3. Rotate your Zerodha password and TOTP secret manually for safety.
4. Resume the manual daily ritual until you're ready to try again.
