@echo off
REM Chain snapshot daemon launcher
REM Auto-registered via Windows Task Scheduler (onlogon trigger)
REM Captures option chain every 5 minutes during market hours

cd /d E:\Projects\options_engine
set PYTHONIOENCODING=utf-8
C:\ProgramData\miniconda3\python.exe -m engine.data.chain_snapshot run --interval-min 5 >> logs\chain_snapshot.log 2>&1
