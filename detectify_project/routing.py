"""
WebSocket routing configuration for real-time video streaming.
"""

from django.urls import re_path
from apps.media_processing.streaming_views import VideoStreamConsumer

websocket_urlpatterns = [
    re_path(r'ws/stream/(?P<case_id>\d+)/(?P<video_id>\d+)/$', VideoStreamConsumer.as_asgi()),
]
