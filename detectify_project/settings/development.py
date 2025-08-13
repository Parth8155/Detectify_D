from .base import *

# Development-specific settings
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Email backend for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "django-db"
# Disable Celery for development - run tasks synchronously
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Additional logging for development
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
