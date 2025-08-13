@echo off
echo =====================================
echo    Detectify - Face Recognition System
echo         Build and Setup Script
echo =====================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/6] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/6] Setting up environment configuration...
if not exist .env (
    copy .env.example .env
    echo Environment file created. Please edit .env if needed.
) else (
    echo Environment file already exists.
)

echo.
echo [3/6] Creating database migrations...
python manage.py makemigrations
if errorlevel 1 (
    echo Error: Failed to create migrations
    pause
    exit /b 1
)

echo.
echo [4/6] Applying database migrations...
python manage.py migrate
if errorlevel 1 (
    echo Error: Failed to apply migrations
    pause
    exit /b 1
)

echo.
echo [5/6] Collecting static files...
python manage.py collectstatic --noinput
if errorlevel 1 (
    echo Warning: Failed to collect static files (this is normal for development)
)

echo.
echo [6/6] Creating superuser account...
echo You can create a superuser account now or skip this step.
set /p create_superuser="Create superuser account? (y/n): "
if /i "%create_superuser%"=="y" (
    python manage.py createsuperuser
)

echo.
echo =====================================
echo     Build completed successfully!
echo =====================================
echo.
echo Next steps:
echo 1. Start the development server:
echo    python manage.py runserver
echo.
echo 2. (Optional) Start Celery worker for async processing:
echo    celery -A detectify_project worker --loglevel=info
echo.
echo 3. (Optional) Start Redis server for Celery:
echo    redis-server
echo.
echo 4. Open http://localhost:8000 in your browser
echo.
echo For production deployment, see README.md
echo.
pause
