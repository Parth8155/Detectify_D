from django.urls import path
from . import views

app_name = 'cases'

urlpatterns = [
    # Dashboard and main views
    path('', views.dashboard, name='dashboard'),
    path('create/', views.create_case, name='create_case'),
    
    # Case management
    path('<int:case_id>/', views.case_detail, name='case_detail'),
    path('<int:case_id>/delete/', views.delete_case, name='delete_case'),
    
    # File uploads
    path('<int:case_id>/upload-suspect/', views.upload_suspect, name='upload_suspect'),
    path('<int:case_id>/upload-video/', views.upload_video, name='upload_video'),
    
    # Processing
    path('<int:case_id>/suspect/<int:suspect_id>/process/', views.process_suspect, name='process_suspect'),
    path('<int:case_id>/video/<int:video_id>/extract-metadata/', views.extract_video_metadata, name='extract_video_metadata'),
    path('<int:case_id>/video/<int:video_id>/process/', views.process_video, name='process_video'),
    path('<int:case_id>/video/<int:video_id>/results/', views.video_results, name='video_results'),
    path('<int:case_id>/video/<int:video_id>/status/', views.video_processing_status, name='video_processing_status'),
        
    # Downloads and media serving
    path('<int:case_id>/download/<int:processed_video_id>/', views.download_processed_video, name='download_processed_video'),
    path('<int:case_id>/suspect/<int:suspect_id>/image/', views.serve_suspect_image, name='serve_suspect_image'),
    path('<int:case_id>/video/<int:video_id>/file/', views.serve_video, name='serve_video'),
    
    # API endpoints
    path('api/<int:case_id>/status/', views.api_case_status, name='api_case_status'),
    
    # Real-time streaming endpoints
    path('<int:case_id>/video/<int:video_id>/stream/', views.stream_video_mjpeg, name='stream_video_mjpeg'),
    path('<int:case_id>/video/<int:video_id>/stream/page/', views.stream_video_page, name='stream_video_page'),
    path('<int:case_id>/video/<int:video_id>/stream/stats/', views.stream_stats, name='stream_stats'),
    path('<int:case_id>/video/<int:video_id>/stream/stop/', views.stop_stream, name='stop_stream'),
    
    # New suspect image serving route
    path('suspect/<int:suspect_id>/image/', views.suspect_image, name='suspect_image'),
]
