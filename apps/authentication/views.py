from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .forms import CustomUserCreationForm, UserProfileForm
from apps.cases.models import Case


def user_login(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('cases:dashboard')
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                next_url = request.GET.get('next', 'cases:dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = AuthenticationForm()
    
    return render(request, 'authentication/login.html', {'form': form})


def user_register(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('cases:dashboard')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('authentication:login')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'authentication/register.html', {'form': form})


@login_required
def user_logout(request):
    """User logout view"""
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('authentication:login')

@login_required
def profile(request):
    """User profile view"""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully!')
            return redirect('authentication:profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserProfileForm(instance=request.user)
    
    # Get user statistics
    user_stats = {
        'total_cases': Case.objects.filter(user=request.user).count(),
        'completed_cases': Case.objects.filter(user=request.user, status='completed').count(),
        'processing_cases': Case.objects.filter(user=request.user, status='processing').count(),
    }
    
    context = {
        'form': form,
        'user_stats': user_stats,
    }
    
    return render(request, 'authentication/profile.html', context)


@require_http_methods(["POST"])
def check_username(request):
    """AJAX endpoint to check if username is available"""
    username = request.POST.get('username', '').strip()
    
    if not username:
        return JsonResponse({'available': False, 'message': 'Username is required'})
    
    if len(username) < 3:
        return JsonResponse({'available': False, 'message': 'Username must be at least 3 characters'})
    
    if User.objects.filter(username=username).exists():
        return JsonResponse({'available': False, 'message': 'Username is already taken'})
    
    return JsonResponse({'available': True, 'message': 'Username is available'})


def password_reset_request(request):
    """Password reset request view"""
    # This would typically handle password reset emails
    # For now, we'll just show a simple message
    if request.method == 'POST':
        email = request.POST.get('email')
        if email:
            # In a real implementation, you'd send a password reset email here
            messages.success(request, 'If an account with that email exists, a password reset link has been sent.')
            return redirect('authentication:login')
        else:
            messages.error(request, 'Please enter a valid email address.')
    
    return render(request, 'authentication/password_reset.html')


def home(request):
    """Landing page view"""
    if request.user.is_authenticated:
        return redirect('cases:dashboard')
    
    return render(request, 'authentication/home.html')
