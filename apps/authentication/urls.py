from django.urls import path
from . import views

app_name = 'authentication'

urlpatterns = [
    # Authentication
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    
    # Profile management
    path('profile/', views.profile, name='profile'),
    
    # Password reset
    path('password-reset/', views.password_reset_request, name='password_reset'),
    
    # AJAX endpoints
    path('check-username/', views.check_username, name='check_username'),
    
    # Home page
    path('', views.home, name='home'),
]
