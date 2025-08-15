#!/usr/bin/env python
"""
Test script to verify async detection saving works correctly
"""

import os
import sys
import django

# Set up Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'detectify_project.settings.development')
django.setup()

def test_detection_saving():
    """Test that detection saving works in sync context"""
    try:
        from apps.cases.models import Case, VideoUpload, SuspectImage, DetectionResult
        from apps.media_processing.realtime_processor import VideoStreamProcessor
        import tempfile
        import json
        import numpy as np
        
        print("=== Testing Detection Saving ===")
        
        # Find a test case and video
        case = Case.objects.first()
        if not case:
            print("❌ No cases found. Please create a case first.")
            return False
            
        video = VideoUpload.objects.filter(case=case).first()
        if not video:
            print("❌ No videos found. Please upload a video first.")
            return False
            
        suspects = SuspectImage.objects.filter(case=case, processed=True)
        if not suspects.exists():
            print("❌ No processed suspects found. Please process suspect images first.")
            return False
        
        print(f"✅ Found case: {case.name}")
        print(f"✅ Found video: {video.video_name}")
        print(f"✅ Found {suspects.count()} processed suspects")
        
        # Prepare suspect data
        suspect_encodings = []
        suspect_mapping = {}
        
        for i, suspect in enumerate(suspects):
            if suspect.face_encoding:
                try:
                    encoding = json.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                    suspect_encodings.append(np.array(encoding))
                    suspect_mapping[i] = suspect
                    print(f"✅ Loaded encoding for suspect {suspect.id}")
                except Exception as e:
                    print(f"❌ Error loading suspect {suspect.id}: {e}")
        
        if not suspect_encodings:
            print("❌ No valid suspect encodings found")
            return False
        
        # Create a test processor with suspect data
        temp_video_path = video.write_temp_file()
        processor = VideoStreamProcessor(
            temp_video_path,
            video_obj=video,
            suspect_encodings=suspect_encodings,
            suspect_mapping=suspect_mapping
        )
        
        # Test adding a mock detection
        test_detection = {
            'video': video,
            'suspect': suspect_mapping[0],
            'timestamp': 10.5,
            'confidence': 0.95,
            'frame_number': 100,
            'bounding_box': (100, 100, 50, 50)
        }
        
        processor.pending_detections.append(test_detection)
        print(f"✅ Added test detection to pending queue")
        
        # Test saving detections
        initial_count = DetectionResult.objects.filter(video=video).count()
        print(f"📊 Initial DetectionResult count: {initial_count}")
        
        saved_count = processor.save_pending_detections()
        print(f"✅ Saved {saved_count} detections")
        
        final_count = DetectionResult.objects.filter(video=video).count()
        print(f"📊 Final DetectionResult count: {final_count}")
        
        if final_count > initial_count:
            print("✅ SUCCESS: Detection saving works correctly!")
            
            # Show the saved detection
            latest = DetectionResult.objects.filter(video=video).latest('created_at')
            print(f"📋 Latest detection: Suspect {latest.suspect.id} at {latest.timestamp}s with confidence {latest.confidence}")
            return True
        else:
            print("❌ FAILURE: No detections were saved")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp file
        try:
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except:
            pass

if __name__ == '__main__':
    success = test_detection_saving()
    if success:
        print("\n🎉 All tests passed! Async detection saving should work correctly.")
    else:
        print("\n⚠️  Tests failed. Please check the error messages above.")
