@echo off
REM Optimized Docker Build Script for Advanced Document Processing System
REM This script builds the optimized Docker container with improved performance

setlocal enabledelayedexpansion

REM Colors for output
set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

echo %BLUE%========================================%NC%
echo %BLUE%  Advanced Doc Selector (CPU-Only)     %NC%
echo %BLUE%         Optimized Version 4.1         %NC%
echo %BLUE%========================================%NC%
echo.

REM Check Docker availability
echo %BLUE%[INFO]%NC% Checking Docker availability...
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker is not installed or not running
    echo Please install Docker Desktop and try again
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker Compose is not available
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Docker is available and running

REM Build optimized image
echo.
echo %BLUE%[INFO]%NC% Building CPU-only Docker image: advanced-doc-selector-cpu:v4.1
echo %YELLOW%This may take 5-10 minutes for the first build...%NC%

docker build -t advanced-doc-selector-cpu:v4.1 .
if errorlevel 1 (
    echo.
    echo %RED%[ERROR]%NC% Docker build failed
    echo Check the error messages above for details
    pause
    exit /b 1
)

echo.
echo %GREEN%[SUCCESS]%NC% Docker image 'advanced-doc-selector-cpu:v4.1' built successfully!

REM Start optimized services
echo.
echo %BLUE%[INFO]%NC% Starting optimized services...
docker-compose up -d

if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to start services
    pause
    exit /b 1
)

echo.
echo %GREEN%[SUCCESS]%NC% Container started successfully!
echo.
echo %BLUE%Next steps:%NC%
echo   1. Add PDF documents to the 'docs' folder
echo   2. Run: docker-compose exec document-processor python run_pipeline.py
echo   3. Check results in the 'outputs' folder
echo.
echo %BLUE%Useful commands:%NC%
echo   • View logs: docker-compose logs -f
echo   • Enter container: docker-compose exec document-processor bash
echo   • Stop services: docker-compose down
echo.
echo %GREEN%Build completed successfully! 🚀%NC%
pause
