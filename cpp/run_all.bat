@echo off
rem ============================================================
rem  run_all.bat - run the 14 steps in order (01 -> 14).
rem  Intermediate binary archives go to .\outputs\*.bin
rem
rem  Usage:
rem    run_all.bat synthetic            (default; use 01_generate_ipd)
rem    run_all.bat caida bignet 16      (real CAIDA/NCTC, subset bignet, K=16)
rem    run_all.bat caida labnet 16      (real CAIDA/NCTC, subset labnet)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=synthetic

rem Steps 02-14 are listed explicitly so the two 01 variants are not both run.
set STEPS=02_preprocess_segment 03_change_rate_symbol 04_multiscale_windows 05_random_matrix_fingerprint 06_hamming_similarity 07_robust_aggregation 08_lsh_candidate 09_threshold_graph 10_gnn_gcn 11_edge_fusion 12_graph_features 13_two_level_classifier 14_evaluation

if /i "%MODE%"=="caida" goto :caida
goto :synthetic

:caida
set SUBSET=%2
if "%SUBSET%"=="" set SUBSET=bignet
set KK=%3
if "%KK%"=="" set KK=16
call :run 01_load_caida %SUBSET% %KK%
goto :after01

:synthetic
call :run 01_generate_ipd
goto :after01

:after01
for %%s in (%STEPS%) do call :run %%s

echo.
echo ============================================================
echo  All 14 steps finished. Intermediate files are in .\outputs\
echo ============================================================
exit /b 0

:run
echo.
echo ############################################################
echo # RUN %~1
echo ############################################################
"%~1.exe" %2 %3 %4
if errorlevel 1 (
    echo.
    echo [RUN FAILED] at %~1
    pause
    exit /b 1
)
exit /b 0
