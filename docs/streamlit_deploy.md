# Streamlit Cloud deployment

The Streamlit dashboard (`streamlit_app.py` + `streamlit_pages/`) reads
directly from Aiven Postgres, so once deployed it works from any device
(phone, other laptops) without your local machine running.

## One-time setup

1. **Push the repo to GitHub** (if not already there).
2. Sign in to https://share.streamlit.io with your GitHub account and grant
   it read access to this repo.
3. Click **New app** -> select this repo -> branch `main` -> main file path
   `streamlit_app.py`.
4. Open **Advanced settings -> Secrets** and paste:

   ```toml
   DATABASE_URL = "postgres://USER:PASS@HOST:5432/DB?sslmode=require"
   ```

   Use the read-only Aiven user if you have one. The dashboard never writes.
   No Kite secrets are needed.
5. Click **Deploy**. First build takes ~3 minutes (installs `requirements.txt`).
6. The app URL is `https://<repo>-<random>.streamlit.app`. Bookmark on your
   phone.

## Local dev

```powershell
cd E:\Projects\options_engine
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# edit .streamlit\secrets.toml with the real DATABASE_URL
C:\ProgramData\miniconda3\python.exe -m streamlit run streamlit_app.py
```

Opens at http://localhost:8501.

Alternatively, export the env var instead of using `secrets.toml`:

```powershell
$env:DATABASE_URL = "postgres://..."
C:\ProgramData\miniconda3\python.exe -m streamlit run streamlit_app.py
```

## Free tier limits

- 1 app per workspace
- 1 GB RAM, 1 CPU
- App is **public by default**. Use **Settings -> Sharing** to add a Google
  SSO allowlist if you want to restrict access.
- Apps go to sleep after ~7 days of zero traffic and wake on first request
  (cold start ~30 s).

## Architecture choices

- **5-min cache** (`@st.cache_data(ttl=300)`) on every Aiven query - keeps DB
  traffic minimal and the UI snappy under the free-tier RAM ceiling.
- **No writes** - the dashboard is strictly read-only. No risk of corrupting
  production data if the cloud instance is compromised.
- **Graceful empty states** - when a table is empty (e.g. before any
  snapshots are written), pages render an informative placeholder rather
  than crashing.
- **Regime + predictive_power pages** are placeholders. Those datasets
  currently live in local parquet (`data/research/...`) and are not yet
  pushed to Aiven. Tracked as future work; new tables `regime_labels` and
  `predictive_power_scores` will let those pages light up.

## Troubleshooting

- **"DATABASE_URL not configured"**: secrets weren't saved. Re-open the
  Streamlit Cloud app settings, paste again, click Save, then **Reboot app**.
- **"could not translate host name"**: Aiven IP allowlist. Streamlit Cloud
  egress IPs are not stable; allow `0.0.0.0/0` in the Aiven service (Aiven's
  Postgres always requires TLS so this is safer than it sounds) or use the
  Aiven VPC peering option.
- **Page is slow on first load**: cold start. Subsequent reads hit the
  5-minute cache.
