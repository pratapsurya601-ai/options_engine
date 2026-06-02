@echo off
REM Research rebuild daemon launcher
REM Auto-registered via Windows Task Scheduler (onlogon trigger)
REM Runs the 6-step research pipeline daily at 16:00 IST

cd /d E:\Projects\options_engine
set PYTHONIOENCODING=utf-8
C:\ProgramData\miniconda3\python.exe -m engine.research.rebuild_daemon run --daily-time 16:00 >> logs\rebuild_daemon.log 2>&1
