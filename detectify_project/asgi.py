"""
ASGI config for detectify_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detectify_project.settings")

# Initialize Django ASGI application early to ensure Django is setup
django_asgi_app = get_asgi_application()

# Import WebSocket routing after Django is setup
from .routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    # For local development, avoid wrapping the WebSocket router with
    # AllowedHostsOriginValidator which can reject connections based on
    # Origin/Host headers and make debugging harder. Remove this wrapper
    # only for local debugging. To re-enable origin checks, wrap the
    # AuthMiddlewareStack with AllowedHostsOriginValidator again.
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
