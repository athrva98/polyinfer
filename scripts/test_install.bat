@echo off
REM Test PolyInfer installation on Windows
REM Usage: scripts\test_install.bat [extras]
REM   scripts\test_install.bat          - Basic install
REM   scripts\test_install.bat cpu      - Install with [cpu]
REM   scripts\test_install.bat all      - Install with [all]

setlocal enabledelayedexpansion

set EXTRAS=%1
set VENV_DIR=test_venv
set PROJECT_DIR=%~dp0..

echo ============================================================
echo PolyInfer Installation Test (Windows)
echo ============================================================
echo Project: %PROJECT_DIR%
echo Extras: %EXTRAS%
echo ============================================================

REM Remove existing venv
if exist "%VENV_DIR%" (
    echo Removing existing virtual environment...
    rmdir /s /q "%VENV_DIR%"
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 goto :error

REM Activate and install
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip -q
if errorlevel 1 goto :error

REM Install polyinfer
if "%EXTRAS%"=="" (
    echo Installing polyinfer...
    pip install -e "%PROJECT_DIR%"
) else (
    echo Installing polyinfer[%EXTRAS%]...
    pip install -e "%PROJECT_DIR%[%EXTRAS%]"
)
if errorlevel 1 goto :error

REM Verify imports
echo.
echo Verifying imports...
python -c "import polyinfer as pi; print(f'Version: {pi.__version__}'); print(f'Backends: {pi.list_backends()}')"
if errorlevel 1 goto :error

REM Install pytest and run tests
echo.
echo Running tests...
pip install pytest -q
python -m pytest "%PROJECT_DIR%\tests" -v --tb=short -x --ignore=tests/test_benchmark.py
set TEST_RESULT=%errorlevel%

REM Cleanup
echo.
echo Cleaning up...
deactivate 2>nul
rmdir /s /q "%VENV_DIR%"

if %TEST_RESULT% neq 0 goto :test_failed

echo.
echo ============================================================
echo INSTALLATION TEST: PASSED
echo ============================================================
exit /b 0

:error
echo.
echo ============================================================
echo INSTALLATION TEST: FAILED (installation error)
echo ============================================================
deactivate 2>nul
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
exit /b 1

:test_failed
echo.
echo ============================================================
echo INSTALLATION TEST: FAILED (tests failed)
echo ============================================================
exit /b 1
