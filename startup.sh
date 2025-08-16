#!/usr/bin/env bash
# Azure App Service startup script

echo "Starting Detectify Django Application..."

# Create necessary directories
mkdir -p /home/site/wwwroot/logs
mkdir -p /home/site/wwwroot/staticfiles
mkdir -p /home/site/wwwroot/media

# Set production environment
export DJANGO_SETTINGS_MODULE=detectify_project.settings.production

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Run database migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Create superuser if it doesn't exist (optional)
echo "Creating superuser..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@detectify.com', 'your_secure_password_here')
    print('Superuser created')
else:
    print('Superuser already exists')
EOF

# Start Gunicorn
echo "Starting Gunicorn server..."
exec gunicorn detectify_project.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
