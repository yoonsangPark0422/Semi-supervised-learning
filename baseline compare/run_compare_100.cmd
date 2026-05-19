@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "OUT=%ROOT%result"

if not exist "%OUT%" mkdir "%OUT%"
cd /d "%ROOT%"

"%PYTHON%" "%ROOT%compare_baselines.py" --out "%OUT%" --methods crest,daso,abc,proposed --epochs 100 --eval-step 1024 --batch-size 32 --mu 7 --num-workers 0 > "%OUT%\runner.stdout.log" 2> "%OUT%\runner.stderr.log"

echo exit_code=%ERRORLEVEL% > "%OUT%\runner.exit.txt"
