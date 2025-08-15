#!/usr/bin/env python
"""
Test script to verify summary video creation functionality
"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings.development')
django.setup()

from apps.cases.models import Case, VideoUpload, ProcessedVideo, DetectionResult

def test_summary_video_creation():
    """Test if summary video creation works properly"""
    print("Testing summary video creation...")
    
    # Check if there are any videos with detections
    videos_with_detections = VideoUpload.objects.filter(
        detections__isnull=False
    ).distinct()
    
    print(f"Found {videos_with_detections.count()} videos with detections")
    
    # Check if there are any processed videos
    processed_videos = ProcessedVideo.objects.all()
    print(f"Found {processed_videos.count()} processed videos")
    
    for processed_video in processed_videos:
        print(f"Processed Video ID: {processed_video.id}")
        print(f"  Original Video: {processed_video.original_video.video_name}")
        print(f"  Total Detections: {processed_video.total_detections}")
        print(f"  Summary Duration: {processed_video.summary_duration}s")
        print(f"  Data Size: {processed_video.processed_size} bytes")
        print(f"  Created: {processed_video.created_at}")
        print("---")
    
    # Check for videos that should have processed videos but don't
    for video in videos_with_detections:
        detection_count = DetectionResult.objects.filter(video=video).count()
        processed_count = ProcessedVideo.objects.filter(original_video=video).count()
        
        print(f"Video: {video.video_name}")
        print(f"  Detections: {detection_count}")
        print(f"  Processed Videos: {processed_count}")
        
        if detection_count > 0 and processed_count == 0:
            print(f"  ⚠️ Video has detections but no summary video!")
        elif processed_count > 0:
            print(f"  ✅ Summary video exists")
        print("---")

if __name__ == "__main__":
    test_summary_video_creation()
