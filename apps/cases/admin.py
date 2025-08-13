from django.contrib import admin
from .models import Case, SuspectImage, VideoUpload, DetectionResult, ProcessedVideo


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'status', 'created_at', 'updated_at']
    list_filter = ['status', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(SuspectImage)
class SuspectImageAdmin(admin.ModelAdmin):
    list_display = ['case', 'uploaded_at', 'processed']
    list_filter = ['processed', 'uploaded_at', 'case__user']
    search_fields = ['case__name']
    readonly_fields = ['uploaded_at', 'face_encoding']
    ordering = ['-uploaded_at']


@admin.register(VideoUpload)
class VideoUploadAdmin(admin.ModelAdmin):
    list_display = ['case', 'duration', 'fps', 'processed', 'uploaded_at']
    list_filter = ['processed', 'uploaded_at', 'case__user']
    search_fields = ['case__name']
    readonly_fields = ['uploaded_at', 'processing_started_at', 'processing_completed_at']
    ordering = ['-uploaded_at']


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ['video', 'suspect', 'timestamp', 'confidence', 'created_at']
    list_filter = ['confidence', 'created_at', 'video__case__user']
    search_fields = ['video__case__name', 'suspect__case__name']
    readonly_fields = ['created_at']
    ordering = ['-created_at']


@admin.register(ProcessedVideo)
class ProcessedVideoAdmin(admin.ModelAdmin):
    list_display = ['case', 'original_video', 'total_detections', 'summary_duration', 'created_at']
    list_filter = ['created_at', 'case__user']
    search_fields = ['case__name']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
