#!/usr/bin/env python
"""
Test script to simulate WebSocket completion flow
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings.development')
django.setup()

from apps.cases.models import VideoUpload, SuspectImage, DetectionResult
from apps.media_processing.streaming_views import _mark_video_processed_sync
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print('=== Testing WebSocket Completion Flow ===')

# Get the latest video that was streamed (should be video 27 or 28)
latest_videos = VideoUpload.objects.filter(processed=True).order_by('-id')[:3]
for video in latest_videos:
    print(f'Video {video.id}: processed={video.processed}, case={video.case.id}, processing_started={video.processing_started_at}')

# Use the latest WebSocket streamed video
test_video = latest_videos[1]  # Use video 27 or 28, not 29 which was manually processed
case_id = test_video.case.id
video_id = test_video.id

print(f'\n=== Testing with Video {video_id} (Case {case_id}) ===')

# Check suspects (simulate WebSocket code)
suspects = SuspectImage.objects.filter(case_id=case_id, processed=True)
suspect_ids = list(suspects.values_list('id', flat=True))
print(f'Found {len(suspect_ids)} processed suspects: {suspect_ids}')

# Check current detection count
current_detections = DetectionResult.objects.filter(video=test_video).count()
print(f'Current DetectionResult count: {current_detections}')

# Test the WebSocket completion flow step by step
print(f'\n=== Step 1: _mark_video_processed_sync ===')
# Reset processed flag to test marking
test_video.processed = False
test_video.save()

# Call the marking function
_mark_video_processed_sync(case_id, video_id)

# Check if it was marked
test_video.refresh_from_db()
print(f'Video marked as processed: {test_video.processed}')

print(f'\n=== Step 2: Batch Processing Trigger ===')
if suspect_ids:
    print(f"WebSocket: Found {len(suspect_ids)} processed suspects for case {case_id}")
    
    try:
        from apps.media_processing.tasks import process_video_task
        
        print(f"WebSocket: Starting batch processing for video {video_id} with suspects {suspect_ids}")
        
        # Test if process_video_task has delay method (Celery)
        if hasattr(process_video_task, 'delay'):
            print("WebSocket: process_video_task has 'delay' method - would call async")
            # Don't actually call it async, just test the logic
            print("WebSocket: Would call: process_video_task.delay(video_id, suspect_ids)")
        else:
            print("WebSocket: Running batch processing synchronously")
            result = process_video_task(video_id, suspect_ids)
            print(f"WebSocket: Batch processing completed: {result}")
            
    except Exception as e:
        print(f"WebSocket: ERROR in batch processing: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"WebSocket: No processed suspects found for case {case_id}, skipping batch processing")

# Check final detection count
final_detections = DetectionResult.objects.filter(video=test_video).count()
print(f'\n=== Final Results ===')
print(f'DetectionResult count after test: {final_detections}')
if final_detections > current_detections:
    print('✅ SUCCESS: WebSocket flow would create DetectionResult records!')
else:
    print('❌ PROBLEM: WebSocket flow does not create DetectionResult records')
