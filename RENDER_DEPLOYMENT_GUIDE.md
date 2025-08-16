# Detectify - Render Deployment Guide

Complete step-by-step guide to deploy Detectify to Render.

## Prerequisites
- GitHub account with your Detectify repository
- Render account (free tier available)

## Step 1: Prepare Your Repository

Make sure these files exist in your repository:
- `requirements.txt` (with all dependencies)
- `detectify_project/settings/production.py` (Render-optimized)
- Your Django project code

## Step 2: Create Render Services

### 2.1 Create PostgreSQL Database
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **PostgreSQL**
3. Fill in details:
   - **Name**: `detectify-db`
   - **Database**: `detectify`
   - **User**: `detectify_user`
   - **Region**: Choose closest to you
   - **Plan**: Free (or paid for production)
4. Click **Create Database**
5. **Save the connection details** - you'll need them for the web service

### 2.2 Create Redis Instance (Optional - for Celery/Channels)
1. Click **New** → **Redis**
2. Fill in details:
   - **Name**: `detectify-redis`
   - **Region**: Same as database
   - **Plan**: Free (or paid)
3. Click **Create Redis**
4. **Save the Redis URL** for environment variables

### 2.3 Create Web Service
1. Click **New** → **Web Service**
2. Connect your GitHub repository
3. Fill in details:
   - **Name**: `detectify-app`
   - **Region**: Same as database
   - **Branch**: `master` (or your main branch)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python manage.py collectstatic --noinput && python manage.py migrate --noinput && gunicorn detectify_project.wsgi:application --bind 0.0.0.0:$PORT`

## Step 3: Configure Environment Variables

In your web service settings, add these environment variables:

### Required Variables
```
DJANGO_SETTINGS_MODULE=detectify_project.settings.production
SECRET_KEY=your-super-secret-key-here-make-it-long-and-random
DEBUG=False
PYTHON_VERSION=3.11.0
```

### Database Variables (from Step 2.1)
```
DATABASE_URL=postgresql://user:password@hostname:port/database
PGDATABASE=detectify
PGUSER=detectify_user
PGPASSWORD=your-db-password
PGHOST=your-db-host
PGPORT=5432
```

### Redis Variables (if using - from Step 2.2)
```
REDIS_URL=redis://red-xxxxx:6379
```

### Optional Variables
```
ALLOWED_HOSTS=your-app-name.onrender.com
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_HOST_USER=your-sendgrid-user
EMAIL_HOST_PASSWORD=your-sendgrid-password
DEFAULT_FROM_EMAIL=noreply@yourapp.com
```

## Step 4: Deploy

1. Click **Create Web Service**
2. Render will automatically:
   - Clone your repository
   - Run the build command
   - Install dependencies
   - Run the start command
   - Deploy your app

## Step 5: Post-Deployment Setup

### 5.1 Create Superuser
1. Go to your service dashboard
2. Click **Shell** tab
3. Run: `python manage.py shell`
4. Create superuser:
```python
from django.contrib.auth import get_user_model
User = get_user_model()
User.objects.create_superuser('admin', 'admin@example.com', 'your-password')
exit()
```

### 5.2 Verify Deployment
- Visit your app URL: `https://your-app-name.onrender.com`
- Check admin panel: `https://your-app-name.onrender.com/admin/`

## Step 6: Set Up Background Worker (For Celery)

If your app uses Celery for background tasks:

1. Click **New** → **Background Worker**
2. Connect same GitHub repository
3. Fill in details:
   - **Name**: `detectify-worker`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `celery -A detectify_project worker --loglevel=info`
4. Add same environment variables as web service
5. Click **Create Background Worker**

## Render Commands Reference

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command (Web Service)
```bash
python manage.py collectstatic --noinput && python manage.py migrate --noinput && gunicorn detectify_project.wsgi:application --bind 0.0.0.0:$PORT
```

### Start Command (Background Worker - if using Celery)
```bash
celery -A detectify_project worker --loglevel=info
```

## Troubleshooting

### Common Issues

1. **Static files not loading**
   - Ensure `collectstatic` runs in start command
   - Check `STATIC_ROOT` and `STATIC_URL` in production.py

2. **Database connection errors**
   - Verify all database environment variables
   - Check PostgreSQL service is running
   - Ensure `sslmode=require` is set

3. **Import errors**
   - Check all dependencies are in `requirements.txt`
   - Verify Python version compatibility

4. **Media files not persisting**
   - Render's filesystem is ephemeral
   - Consider using Cloudinary for media storage

### Logs and Monitoring
- View logs in Render Dashboard → Your Service → Logs
- Set up log-based alerts in Render
- Monitor service health and performance

## Environment Variables Template

Create a `.env.render` file (don't commit to repo) with:
```bash
DJANGO_SETTINGS_MODULE=detectify_project.settings.production
SECRET_KEY=your-secret-key
DEBUG=False
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
ALLOWED_HOSTS=your-app.onrender.com
```

## Cost Considerations

### Free Tier Limits
- Web Service: 512MB RAM, sleeps after 15min inactivity
- PostgreSQL: 1GB storage, 97 hours/month
- Redis: 25MB storage

### Upgrading
- Consider paid plans for production apps
- Background workers require paid plan
- More resources and always-on availability

## Security Checklist

- [ ] Set strong `SECRET_KEY`
- [ ] Set `DEBUG=False`
- [ ] Configure proper `ALLOWED_HOSTS`
- [ ] Use environment variables for sensitive data
- [ ] Enable SSL (automatic on Render)
- [ ] Set up proper database user permissions

## Monitoring and Maintenance

- Monitor service health in Render dashboard
- Set up database backups
- Monitor resource usage
- Keep dependencies updated
- Monitor error logs for issues
