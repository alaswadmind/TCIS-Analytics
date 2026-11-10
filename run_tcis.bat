@echo off
chcp 65001 > nul
title TCIS - Telecom Customer Intelligence System
echo =================================================================
echo   TCIS - Telecom Customer Intelligence System
echo   Launching Enterprise AI Analytics Platform
echo =================================================================
cd /d "%~dp0"
python run_server.py
pause
