# Running the Research Rebuild Daemon as a Windows Service (NSSM)

The daemon (`engine.research.rebuild_daemon`) runs the 6 research modules
once per day at 16:00 IST. For unattended operation across reboots, install
it as a Windows service via [NSSM](https://nssm.cc/).

## Prerequisites

- NSSM installed and on PATH (`nssm --version` should work).
- `C:\ProgramData\miniconda3\python.exe` available with project deps.
- Project checked out at a stable path, e.g. `E:\Projects\options_engine`.

## Install

Run an Administrator PowerShell:

```powershell
nssm install OptionsResearchDaemon `
    "C:\ProgramData\miniconda3\python.exe" `
    "-m engine.research.rebuild_daemon run --daily-time 16:00"

nssm set OptionsResearchDaemon AppDirectory "E:\Projects\options_engine"
nssm set OptionsResearchDaemon AppStdout    "E:\Projects\options_engine\logs\research_daemon.out.log"
nssm set OptionsResearchDaemon AppStderr    "E:\Projects\options_engine\logs\research_daemon.err.log"
nssm set OptionsResearchDaemon Start        SERVICE_AUTO_START

nssm start OptionsResearchDaemon
```

## Verify

```powershell
nssm status OptionsResearchDaemon
Get-Content "E:\Projects\options_engine\logs\research_daemon.out.log" -Tail 20
```

After the first 16:00 IST trigger, you should see
`data/research/rebuild_status.json` updated and the `/research` dashboard
header reflect the new run.

## Update / Remove

```powershell
nssm restart OptionsResearchDaemon       # after code changes
nssm stop    OptionsResearchDaemon
nssm remove  OptionsResearchDaemon confirm
```

## Alternative: Task Scheduler

If you'd rather not install NSSM:

```cmd
schtasks /create /tn "OptionsResearchDaemon" ^
    /tr "C:\ProgramData\miniconda3\python.exe -m engine.research.rebuild_daemon run" ^
    /sc onlogon /f
```

The daemon itself handles the daily timing, so `onlogon` is enough -- you
don't need a per-day scheduled trigger.
