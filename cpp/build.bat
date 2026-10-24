@echo off
rem ============================================================
rem  build.bat - compile all 14 numbered steps of the QuadFP-Cascade
rem  C++11 port (g++ 4.9.2 / MinGW, no third-party dependencies).
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo [build] C++11 (g++ -std=c++11 -O2 -static), compiling numbered steps ...
for %%f in (??_*.cpp) do (
    echo   compiling %%f ...
    g++ -std=c++11 -O2 -static -o "%%~nf.exe" "%%f"
    if errorlevel 1 (
        echo.
        echo [BUILD FAILED] at %%f
        pause
        exit /b 1
    )
)
echo.
echo [build OK] all executables generated next to the sources.
