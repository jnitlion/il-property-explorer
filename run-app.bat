@echo off
rem Launch the Ionic-Liquid Property Explorer dashboard.
cd /d "%~dp0"
"C:\Users\jnitlion\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
pause
