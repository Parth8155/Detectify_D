#!/usr/bin/env python
"""
Quick script to check DetectionResult records and video processing status
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings.development')
django.setup()

from apps.cases.models import DetectionResult, VideoUpload

print('=== Database Check ===')
print('Total DetectionResult records:', DetectionResult.objects.count())
print('\nRecent videos:')
for v in VideoUpload.objects.order_by('-id')[:5]:
    print(f'Video {v.id}: processed={v.processed}, processing_started={v.processing_started_at}, completed={v.processing_completed_at}')

print('\nDetectionResults for recent videos:')
for v in VideoUpload.objects.order_by('-id')[:5]:
    detection_count = DetectionResult.objects.filter(video=v).count()
    print(f'Video {v.id} detections: {detection_count}')
    if detection_count > 0:
        detections = DetectionResult.objects.filter(video=v)[:3]
        for d in detections:
            print(f'  - Detection at {d.timestamp:.2f}s, confidence={d.confidence:.3f}, suspect={d.suspect.id}')

# Check if process_video_task is being called after streaming
print('\n=== Checking Latest Video Details ===')
latest_video = VideoUpload.objects.order_by('-id').first()
if latest_video:
    print(f'Latest video {latest_video.id}:')
    print(f'  - File size: {latest_video.video_size} bytes')
    print(f'  - Processed: {latest_video.processed}')
    print(f'  - Processing started: {latest_video.processing_started_at}')
    print(f'  - Processing completed: {latest_video.processing_completed_at}')
    print(f'  - Duration: {latest_video.duration}')
    print(f'  - Case: {latest_video.case.id}')
    
    # Check suspects in the same case
    suspects = latest_video.case.suspect_images.all()
    print(f'  - Suspects in case: {suspects.count()}')
    for suspect in suspects:
        print(f'    - Suspect {suspect.id}: processed={suspect.processed}, has_encoding={bool(suspect.face_encoding)}')
