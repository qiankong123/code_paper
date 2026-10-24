@echo off
rem ============================================================
rem  run_all.bat - run the 14 steps in order (01 -> 14).
rem  Intermediate binary archives go to .\outputs\*.bin
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

for %%f in (??_*.exe) do (
    echo.
    echo ############################################################
    echo # RUN %%~nf
    echo ############################################################
    "%%f"
    if errorlevel 1 (
        echo.
        echo [RUN FAILED] at %%f
        pause
        exit /b 1
    )
)
echo.
echo ============================================================
echo  All 14 steps finished. Intermediate files are in .\outputs\
echo ============================================================
