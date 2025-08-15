#!/usr/bin/env python
"""
Test script to verify batch processing works correctly
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings.development')
django.setup()

from apps.cases.models import VideoUpload, SuspectImage, DetectionResult
from apps.media_processing.tasks import process_video_task

print('=== Testing Batch Processing ===')

# Get the latest video that was streamed but has no DetectionResults
latest_video = VideoUpload.objects.filter(processed=True).order_by('-id').first()
print(f'Testing with video {latest_video.id} (case {latest_video.case.id})')

# Check suspects
suspects = SuspectImage.objects.filter(case_id=latest_video.case.id, processed=True)
suspect_ids = list(suspects.values_list('id', flat=True))
print(f'Found {len(suspect_ids)} processed suspects: {suspect_ids}')

# Check current detection count
current_detections = DetectionResult.objects.filter(video=latest_video).count()
print(f'Current DetectionResult count for video {latest_video.id}: {current_detections}')

if len(suspect_ids) > 0:
    print(f'\n=== Running process_video_task manually ===')
    try:
        # Reset video processing flags to test the task
        latest_video.processed = False
        latest_video.processing_started_at = None
        latest_video.processing_completed_at = None
        latest_video.save()
        
        # Run the batch processing task
        result = process_video_task(latest_video.id, suspect_ids)
        print(f'Task result: {result}')
        
        # Check detection count after processing
        new_detections = DetectionResult.objects.filter(video=latest_video).count()
        print(f'DetectionResult count after processing: {new_detections}')
        
        if new_detections > current_detections:
            print('✅ SUCCESS: Batch processing created new DetectionResult records!')
            # Show some sample detections
            detections = DetectionResult.objects.filter(video=latest_video)[:5]
            for d in detections:
                print(f'  - Detection at {d.timestamp:.2f}s, confidence={d.confidence:.3f}, suspect={d.suspect.id}')
        else:
            print('❌ PROBLEM: No new DetectionResult records created')
            
    except Exception as e:
        print(f'❌ ERROR running batch processing: {e}')
        import traceback
        traceback.print_exc()
else:
    print('❌ No processed suspects found - cannot test batch processing')
