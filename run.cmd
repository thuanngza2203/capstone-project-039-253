@echo off
rem Bat ca he thong bang Docker: nhap dup file nay (script that o run.ps1).
rem Thue Vast moi: dan lenh SSH cua Vast khi duoc hoi, hoac: run.cmd ssh -p 59921 root@92.180.27.84
cd /d "%~dp0"
set "VAST_SSH_INPUT=%*"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 pause
