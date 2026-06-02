# Options Engine - Operational Status Check
# Run: powershell -ExecutionPolicy Bypass -File E:\Projects\options_engine\scripts\status.ps1

$ErrorActionPreference = "SilentlyContinue"
$ProjectRoot = "E:\Projects\options_engine"

function Write-Section { param([string]$Title) Write-Host ""; Write-Host "==== $Title ====" -ForegroundColor Cyan }
function Write-OK   { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-Bad  { param([string]$msg) Write-Host "  [XX] $msg" -ForegroundColor Red }
function Write-Info { param([string]$msg) Write-Host "  $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host "  OPTIONS ENGINE - OPERATIONAL STATUS"           -ForegroundColor White
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "  $now"                                          -ForegroundColor White
Write-Host "================================================" -ForegroundColor White

# ---------- 1. Python processes ----------
Write-Section "Running Python Processes"
$pyProcs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'")
if ($pyProcs.Count -eq 0) {
    Write-Bad "NO python.exe processes running"
} else {
    foreach ($p in $pyProcs) {
        $cmd = [string]$p.CommandLine
        $age = ((Get-Date) - $p.CreationDate).ToString("hh\:mm\:ss")
        $tag = "unknown"
        if     ($cmd -match "chain_snapshot")                { $tag = "chain_snapshot" }
        elseif ($cmd -match "rebuild_daemon")                { $tag = "rebuild_daemon" }
        elseif ($cmd -match "engine\.watcher.*htf_naked")    { $tag = "watcher htf_naked" }
        elseif ($cmd -match "engine\.watcher.*panic_bounce") { $tag = "watcher panic_bounce" }
        elseif ($cmd -match "engine\.watcher")               { $tag = "watcher" }
        elseif ($cmd -match "engine\.web\.app")              { $tag = "dashboard" }
        $line = "PID {0,-6} age {1}  {2}" -f $p.ProcessId, $age, $tag
        Write-OK $line
    }
}

# ---------- 2. Critical daemons present? ----------
Write-Section "Critical Services"

$chainRunning   = @($pyProcs | Where-Object { $_.CommandLine -match "chain_snapshot" })
$rebuildRunning = @($pyProcs | Where-Object { $_.CommandLine -match "rebuild_daemon" })
$dashRunning    = @($pyProcs | Where-Object { $_.CommandLine -match "engine\.web\.app" })
$watcherRunning = @($pyProcs | Where-Object { $_.CommandLine -match "engine\.watcher" })

if ($chainRunning.Count -gt 0)   { Write-OK  "chain_snapshot daemon: running (PID $($chainRunning[0].ProcessId))" }
else                             { Write-Bad "chain_snapshot daemon: NOT RUNNING" }

if ($rebuildRunning.Count -gt 0) { Write-OK  "rebuild_daemon: running (PID $($rebuildRunning[0].ProcessId))" }
else                             { Write-Bad "rebuild_daemon: NOT RUNNING" }

if ($dashRunning.Count -gt 0)    { Write-OK  "Flask dashboard: running (PID $($dashRunning[0].ProcessId))" }
else                             { Write-Warn "Flask dashboard: not running" }

if ($watcherRunning.Count -gt 0) {
    Write-OK "watcher(s): $($watcherRunning.Count) running"
    foreach ($w in $watcherRunning) {
        if ($w.CommandLine -match "--rule\s+(\S+)") {
            Write-Info "    rule: $($Matches[1])"
        }
    }
} else { Write-Warn "No watcher running (not needed off-market-hours)" }

# ---------- 3. Scheduled tasks ----------
Write-Section "Scheduled Tasks"

foreach ($taskName in @("OptionsEngineChainSnapshot", "OptionsEngineRebuildDaemon")) {
    $out = schtasks /Query /TN $taskName /FO LIST 2>$null
    if ($LASTEXITCODE -eq 0) {
        $status = ($out | Select-String "Status:").Line -replace ".*Status:\s+", ""
        $last   = ($out | Select-String "Last Run Time:").Line -replace ".*Last Run Time:\s+", ""
        $result = ($out | Select-String "Last Result:").Line -replace ".*Last Result:\s+", ""
        Write-OK "$taskName"
        Write-Info "    Status: $status"
        Write-Info "    Last Run: $last"
        Write-Info "    Last Result: $result"
    } else {
        Write-Bad "$taskName - NOT REGISTERED"
    }
}

# ---------- 4. Recent log activity ----------
Write-Section "Recent Daemon Log Activity"

$logs = [ordered]@{
    "chain_snapshot.log"   = "$ProjectRoot\logs\chain_snapshot.log"
    "rebuild_daemon.log"   = "$ProjectRoot\logs\rebuild_daemon.log"
    "watcher_2min.log"     = "$ProjectRoot\logs\watcher_2min.log"
    "watcher_15min.log"    = "$ProjectRoot\logs\watcher_15min.log"
    "dashboard.log"        = "$ProjectRoot\logs\dashboard.log"
}
foreach ($name in $logs.Keys) {
    $path = $logs[$name]
    if (Test-Path $path) {
        $f = Get-Item $path
        $ageMin = [int]((Get-Date) - $f.LastWriteTime).TotalMinutes
        $tail = (Get-Content $path -Tail 1) -join " "
        if ($tail.Length -gt 80) { $tail = $tail.Substring(0, 77) + "..." }
        $line = "{0,-22} {1,6} bytes, last write {2,5}m ago" -f $name, $f.Length, $ageMin
        Write-OK $line
        if ($tail) { Write-Info "    last: $tail" }
    } else {
        Write-Warn "$name - does not exist"
    }
}

# ---------- 5. Chain snapshot freshness ----------
Write-Section "Chain Snapshot Data"

$snapRoot = "$ProjectRoot\data\chain_snapshots\NIFTY"
if (Test-Path $snapRoot) {
    $dayDirs = @(Get-ChildItem $snapRoot -Directory | Sort-Object Name -Descending)
    $totalSnaps = 0
    foreach ($d in $dayDirs) {
        $totalSnaps += @(Get-ChildItem $d.FullName -File).Count
    }
    Write-OK "Total: $totalSnaps snapshots across $($dayDirs.Count) days"
    if ($dayDirs.Count -gt 0) {
        $latest = $dayDirs[0]
        $latestSnaps = @(Get-ChildItem $latest.FullName -File | Sort-Object Name -Descending)
        $latestCount = $latestSnaps.Count
        $lastSnapTime = "n/a"
        if ($latestSnaps.Count -gt 0) {
            $lastSnapTime = $latestSnaps[0].LastWriteTime.ToString("HH:mm:ss")
        }
        $msg = "Most recent day: {0} - {1} snapshots, last at {2}" -f $latest.Name, $latestCount, $lastSnapTime
        Write-Info $msg
    }
} else {
    Write-Bad "No chain snapshots directory found"
}

# ---------- 6. Last rebuild status ----------
Write-Section "Last Research Pipeline Rebuild"

$statusPath = "$ProjectRoot\data\research\rebuild_status.json"
if (Test-Path $statusPath) {
    $s = Get-Content $statusPath -Raw | ConvertFrom-Json
    $okSteps = @($s.steps | Where-Object { $_.status -eq "OK" }).Count
    $totSteps = @($s.steps).Count
    $summary = "Overall: {0}  ({1}/{2} steps OK)" -f $s.overall_status, $okSteps, $totSteps
    if     ($s.overall_status -eq "OK")      { Write-OK   $summary }
    elseif ($s.overall_status -eq "PARTIAL") { Write-Warn $summary }
    else                                     { Write-Bad  $summary }
    Write-Info "Started:   $($s.run_started_at)"
    Write-Info "Completed: $($s.run_completed_at)"
    Write-Info "Next run:  $($s.next_run_at)"
    foreach ($step in $s.steps) {
        $st = if ($step.status -eq "OK") { "[OK]" } else { "[XX]" }
        $stepLine = "    {0} {1,-18} {2,5:F1}s" -f $st, $step.name, $step.duration_s
        Write-Info $stepLine
    }
} else {
    Write-Warn "No rebuild_status.json yet (rebuild_daemon has not completed a cycle)"
}

# ---------- 7. Paper trade activity ----------
Write-Section "Paper Trading"

$paperPath = "$ProjectRoot\logs\paper.jsonl"
if (Test-Path $paperPath) {
    $lines = @(Get-Content $paperPath)
    $openCount = 0
    $closedCount = 0
    foreach ($l in $lines) {
        try {
            $j = $l | ConvertFrom-Json
            if     ($j.event -eq "open")  { $openCount++ }
            elseif ($j.event -eq "close") { $closedCount++ }
        } catch {}
    }
    $netOpen = $openCount - $closedCount
    Write-OK "Total opens: $openCount, closes: $closedCount, currently open: $netOpen"
    $lastLines = $lines | Select-Object -Last 3
    Write-Info "Last 3 paper events:"
    foreach ($l in $lastLines) {
        try {
            $j = $l | ConvertFrom-Json
            $entryPrem = if ($null -ne $j.entry_premium) { $j.entry_premium } else { "n/a" }
            $rule = if ($null -ne $j.rule) { $j.rule } else { "?" }
            $line = "    {0} {1} {2} @ {3}" -f $j.ts, $j.event, $rule, $entryPrem
            Write-Info $line
        } catch {
            Write-Info "    (unparseable line)"
        }
    }
} else {
    Write-Warn "No paper.jsonl yet"
}

# ---------- 8. Dashboard reachable? ----------
Write-Section "Dashboard Health Check"

try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:5050/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    Write-OK "Dashboard responding: HTTP $($r.StatusCode)"
    Write-Info "URL:       http://127.0.0.1:5050"
    Write-Info "Research:  http://127.0.0.1:5050/research"
    Write-Info "Briefing:  http://127.0.0.1:5050/briefing"
} catch {
    Write-Warn "Dashboard not reachable (Flask not running, or wrong port)"
}

Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host "  Status check complete." -ForegroundColor White
Write-Host "================================================" -ForegroundColor White
Write-Host ""
