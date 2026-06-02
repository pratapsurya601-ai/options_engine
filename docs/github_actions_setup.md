# GitHub Actions Setup — Free 24/7 Chain Snapshot Capture

This document explains how to set up `.github/workflows/chain_snapshot.yml` so
GitHub Actions captures option chain snapshots every 5 min during Indian market
hours and writes them to Aiven Postgres. **Cost: free** (public repo = unlimited
Actions minutes; private = ~250 min/month vs 2000 free quota).

## One-time setup

### 1. Push repo to GitHub
Already done — see https://github.com/pratapsurya601-ai/options_engine

### 2. Apply the Aiven schema
Already done. Verify with:
```
psql "$DATABASE_URL" -c "\dt"
```
You should see `option_chain_snapshots` listed.

### 3. Add GitHub repository secrets

Open https://github.com/pratapsurya601-ai/options_engine/settings/secrets/actions
and add four secrets:

| Name | Value | Where to get it |
|---|---|---|
| `DATABASE_URL` | `postgres://avnadmin:...@pg-...aivencloud.com:PORT/defaultdb?sslmode=require` | Aiven console → Service → Service URI |
| `KITE_API_KEY` | `tb1qt4j8mruazarv` | developers.kite.trade → Your apps |
| `KITE_API_SECRET` | (32-char string) | Same page, "Show API secret" |
| `KITE_ACCESS_TOKEN` | (daily token) | Refresh procedure below |

### 4. Verify Actions is enabled
Repo Settings → Actions → General → Allow all actions → Save.

### 5. Test the workflow manually
Go to https://github.com/pratapsurya601-ai/options_engine/actions
→ "Chain Snapshot Capture" → Run workflow → main branch → Run workflow.

Within ~1 min you should see a green checkmark. Then verify Aiven received data:
```
python -m engine.data.chain_snapshot aiven-status
```

## The Kite Token Rotation Problem

**Kite access tokens expire at 6:00 AM IST daily.** The GitHub Action will
fail with auth errors every morning until you refresh the token. This is the
single biggest operational gotcha.

### Daily refresh procedure (5 min, before market open)

1. Visit your Kite login URL (open in browser, log in):
   `https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY&v=3`
2. After login, you'll be redirected to your callback URL with a
   `request_token` query parameter (e.g.
   `http://127.0.0.1:5050/kite/callback?request_token=ABC123&action=login&status=success`)
3. Copy that `request_token` value
4. Exchange for an access token locally:
   ```
   python -m engine.data.kite_source login --request-token ABC123
   ```
   This saves to `~/.kite_token.json`. Print the new token:
   ```
   python -c "import json; print(json.load(open('C:/Users/vinit/.kite_token.json'))['access_token'])"
   ```
5. Update the GitHub secret (via `gh` CLI, or web UI):
   ```
   gh secret set KITE_ACCESS_TOKEN --body "<new-access-token>"
   ```

### Better: automate it
You can script steps 1-5 with the dashboard's `/kite/login` + `/kite/callback`
routes, then a tiny `gh secret set` call. Future enhancement.

## Schedule

The workflow runs on three cron lines that together cover Indian market hours:

| Cron (UTC) | Translates to (IST) | Coverage |
|---|---|---|
| `45,50,55 3 * * 1-5` | 9:15, 9:20, 9:25 IST | Opening |
| `*/5 4-9 * * 1-5` | 9:30 → 15:25 IST | Main session |
| `0 10 * * 1-5` | 15:30 IST | Closing snapshot |

Total: 75 snapshots per trading day. GitHub may add 5-15 min jitter on busy
periods — for our weekly research analysis this doesn't matter.

## Monitoring

- **Last 20 runs:** `gh run list --workflow chain_snapshot.yml --limit 20`
- **Latest failure logs:** `gh run view --log-failed`
- **Aiven row count:** `python -m engine.data.chain_snapshot aiven-status`

## Troubleshooting

### "TokenException: Token is invalid or has expired"
→ Kite token rotation needed. See procedure above.

### "could not translate host name"
→ Aiven service paused. Go to console.aiven.io, click Resume.

### "0 rows written"
→ Either market closed (off-hours run — harmless) or chain was empty
(weekend / holiday).

### "fatal: could not read from remote repository"
→ Repo permissions issue. Settings → Actions → Workflow permissions →
Read and write permissions.

### Workflow not triggering on schedule
→ GitHub disables cron on repos with no commits for 60 days. Push any
commit to re-enable. Also check Actions tab → "Enable workflow".

## Cost summary (private repo)

- 75 runs/day × 1 min/run × 5 weekdays/week = ~25 hours/month
- Free quota: 2,000 minutes/month
- Headroom: ~6x

Public repo: completely free, unlimited minutes.

## Disabling

To stop the workflow:
- Web UI: Actions tab → "Chain Snapshot Capture" → "..." → Disable workflow
- CLI: `gh workflow disable chain_snapshot.yml`

Local laptop daemon (if still running) continues independently.
