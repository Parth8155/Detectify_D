# Detectify - Face Recognition System

A Django-based face recognition system for automated face detection and identification in multimedia content.

## Project Overview

Detectify allows users to upload suspect images and videos, then processes the media to identify faces, extract timestamps, and generate condensed video summaries containing only relevant segments.

## Features

- **User Authentication**: Secure registration, login, and profile management
- **Case Management**: Organize investigations with multiple suspects and videos
- **Face Recognition**: AI-powered face detection using DeepFace
- **Video Processing**: Automated video analysis with timestamp extraction
- **Summary Generation**: Create condensed videos with detection highlights
- **Async Processing**: Background task processing with Celery

## Technology Stack

- **Backend**: Django 4.2+ with Python 3.10+
- **Database**: SQLite (development)
- **Face Recognition**: DeepFace with custom implementation
- **Video Processing**: MoviePy and OpenCV
- **Task Queue**: Celery with Redis
- **Frontend**: Bootstrap 5 with Django Templates
- **File Storage**: Django's secure media handling

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Redis server (for Celery)
- Git

### 1. Clone and Setup

```bash
# Clone the repository
cd "c:\Users\asus\OneDrive\Pictures\CE\SEM-4\Projects\Detectify"

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
copy .env.example .env

# Edit .env file with your configurations
# Note: Default settings work for development
```

### 3. Database Setup

```bash
# Create and apply migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser account
python manage.py createsuperuser
```

### 4. Static Files

```bash
# Collect static files
python manage.py collectstatic --noinput
```

### 5. Start Development Server

```bash
# Start Django development server
python manage.py runserver

# In a new terminal, start Celery worker (optional for async processing)
celery -A detectify_project worker --loglevel=info

# In another terminal, start Redis server
redis-server
```

## Quick Start

1. Visit `http://localhost:8000` to access the application
2. Register a new account or login
3. Create a new case from the dashboard
4. Upload suspect images for the case
5. Upload videos to analyze
6. Start processing to detect suspects in videos
7. Download summary videos with detection highlights

## Project Structure

```
detectify/
├── detectify_project/          # Django project settings
│   ├── settings/              # Environment-specific settings
│   ├── urls.py               # Main URL configuration
│   └── celery.py             # Celery configuration
├── apps/                     # Django applications
│   ├── authentication/       # User management
│   ├── cases/                # Case and file management
│   ├── media_processing/     # Face recognition and video processing
│   └── core/                 # Shared utilities
├── templates/                # HTML templates
├── static/                   # Static files (CSS, JS, images)
├── media/                    # User uploaded files
└── requirements.txt          # Python dependencies
```

## Supported File Formats

### Images
- PNG
- JPG/JPEG

### Videos
- MP4
- AVI
- MKV
- MOV

**Maximum file size**: 500MB per upload

## API Endpoints (Optional)

The system includes REST API endpoints for programmatic access:

- `GET /api/cases/` - List cases
- `POST /api/cases/` - Create case
- `POST /api/cases/{id}/upload/` - Upload media
- `POST /api/cases/{id}/process/` - Start processing
- `GET /api/cases/{id}/results/` - Get results

## Development

### Running Tests

```bash
python manage.py test
```

### Code Formatting

```bash
black .
flake8 .
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

## Configuration

### Face Recognition Settings

```python
# In settings.py or .env
FACE_RECOGNITION_CONFIDENCE_THRESHOLD = 0.85  # Detection confidence
FRAME_SAMPLING_INTERVAL = 0.25               # Frame sampling rate (seconds)
VIDEO_CLIP_BUFFER_BEFORE = 2                 # Buffer before detection (seconds)
VIDEO_CLIP_BUFFER_AFTER = 3                  # Buffer after detection (seconds)
```

### Performance Tuning

- Adjust frame sampling interval for speed vs accuracy trade-off
- Configure Celery workers based on system resources
- Use Redis for session storage in production
- Enable GPU acceleration for face recognition if available

## Deployment

### Production Setup

1. Set `DEBUG=False` in settings
2. Configure production database (PostgreSQL recommended)
3. Set up proper web server (nginx + gunicorn)
4. Configure Celery with production broker
5. Set up proper logging and monitoring
6. Configure SSL certificates

### Docker Deployment (Optional)

```bash
# Build Docker image
docker build -t detectify .

# Run with docker-compose
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Errors**: Run migrations and check database connection
3. **File Upload Issues**: Check media directory permissions
4. **Celery Tasks Not Running**: Ensure Redis is running and Celery worker is started
5. **Face Recognition Errors**: Install required system dependencies for OpenCV

### Getting Help

- Check the Django logs for detailed error messages
- Ensure all required services (Redis) are running
- Verify file permissions for media directory
- Check Python path and virtual environment activation

## License

This project is for educational purposes. Please respect privacy and legal requirements when using face recognition technology.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Changelog

### Version 1.0.0
- Initial release with core functionality
- Face detection and recognition
- Video processing and summary generation
- User authentication and case management
- Async task processing with Celery
