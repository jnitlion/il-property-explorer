@echo off
rem Resume/continue the ILThermo data harvest. Safe to run repeatedly:
rem already-downloaded datasets are skipped automatically.
cd /d "%~dp0"
"C:\Users\jnitlion\AppData\Local\Programs\Python\Python312\python.exe" src\harvest.py
pause
