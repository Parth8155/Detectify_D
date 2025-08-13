"""
Celery configuration for Detectify project
"""

import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings')

app = Celery('detectify')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Optional: Configure task routes
app.conf.task_routes = {
    'apps.media_processing.tasks.process_video_task': {'queue': 'video_processing'},
    'apps.media_processing.tasks.process_suspect_image_task': {'queue': 'image_processing'},
    'apps.media_processing.tasks.cleanup_old_files_task': {'queue': 'maintenance'},
}

# Configure task result expiration
app.conf.result_expires = 3600  # 1 hour

# Configure task retry settings
app.conf.task_default_retry_delay = 60  # 1 minute
app.conf.task_max_retries = 3

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
