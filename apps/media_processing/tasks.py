"""
Celery tasks for asynchronous video processing in Detectify
"""

from celery import shared_task
from django.core.files.base import ContentFile
from django.utils import timezone
from typing import List
import os
import tempfile
import json
import traceback
import numpy as np

from apps.cases.models import VideoUpload, SuspectImage, DetectionResult, ProcessedVideo, Case
from .video_processor import VideoProcessor
from .deepface_client import DeepFaceClient
from django.conf import settings
from PIL import Image
import cv2

@shared_task(bind=True, max_retries=3)
def process_video_task(self, video_id: int, suspect_ids: List[int]):
    """
    Process video to detect suspect faces
    
    Args:
        video_id (int): ID of VideoUpload instance
        suspect_ids (List[int]): List of SuspectImage IDs to search for
        
    Returns:
        Dict: Processing results
    """
    try:
        # Get video and suspects
        video = VideoUpload.objects.get(id=video_id)
        suspects = SuspectImage.objects.filter(id__in=suspect_ids)
        
        # Update processing status
        video.processing_started_at = timezone.now()
        video.save()
        
        case = video.case
        case.status = 'processing'
        case.save()
        
        # Initialize processors
        video_processor = VideoProcessor()
        face_client = DeepFaceClient()
        
        # Extract suspect encodings
        suspect_encodings = []
        suspect_mapping = {}
        
        for i, suspect in enumerate(suspects):
            if suspect.face_encoding:
                encoding = json.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                suspect_encodings.append(encoding)
                suspect_mapping[i] = suspect
            elif suspect.image_data:
                # Generate encoding if not exists using database-stored image
                try:
                    # Create temporary file from database-stored image
                    temp_dir = tempfile.gettempdir()
                    temp_image_path = os.path.join(temp_dir, f"temp_suspect_{suspect.id}.{suspect.image_type}")
                    
                    # Write image data to temporary file
                    with open(temp_image_path, 'wb') as f:
                        f.write(suspect.get_image_data())
                    
                    try:
                        encoding = face_client.extract_features(temp_image_path)
                        suspect.face_encoding = encoding.tolist()
                        suspect.processed = True
                        suspect.save()
                        
                        suspect_encodings.append(encoding)
                        suspect_mapping[i] = suspect
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                except Exception as e:
                    print(f"Error processing suspect {suspect.id}: {str(e)}")
        
        if not suspect_encodings:
            raise ValueError("No valid suspect encodings found")
        
        # Create temporary file from database-stored video
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"temp_video_{video.id}.{video.video_type}")
        
        # Write video data to temporary file
        with open(video_path, 'wb') as f:
            f.write(video.get_video_data())
        
        # Get video metadata for progress calculation
        metadata = video_processor.extract_video_metadata(video_path)
        total_frames = metadata.get('frame_count', 0)
        fps = metadata.get('fps', 30)
        
        # Create progress callback
        def progress_callback(current_frame, frame_timestamp, detections_found):
            progress_percent = (current_frame / total_frames * 100) if total_frames > 0 else 0
            
            # Only update task state if we're running as a real Celery task
            # Skip progress updates when running synchronously (CELERY_TASK_ALWAYS_EAGER=True)
            try:
                if not getattr(settings, 'CELERY_TASK_ALWAYS_EAGER', False) and self.request.id:
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current_frame': current_frame,
                            'total_frames': total_frames,
                            'progress': round(progress_percent, 1),
                            'timestamp': frame_timestamp,
                            'detections_found': detections_found,
                            'fps': fps,
                            'status': 'processing'
                        }
                    )
                else:
                    # For synchronous execution, just print progress
                    print(f"Processing frame {current_frame}/{total_frames} ({progress_percent:.1f}%) - {detections_found} detections at {frame_timestamp:.2f}s")
            except Exception as e:
                # Fallback: just print progress if task state update fails
                print(f"Processing frame {current_frame}/{total_frames} ({progress_percent:.1f}%) - {detections_found} detections at {frame_timestamp:.2f}s")
        
        detections = video_processor.analyze_video_for_suspects(
            video_path, 
            suspect_encodings, 
            progress_callback=progress_callback
        )
        # Save detection results
        detection_objects = []
        for detection in detections:
            suspect = suspect_mapping.get(detection['suspect_index'])
            if suspect:
                # Convert bounding_box values to Python int for JSON serialization
                bbox = tuple(int(v) for v in detection['bounding_box'])
                detection_obj = DetectionResult(
                    video=video,
                    suspect=suspect,
                    timestamp=detection['timestamp'],
                    confidence=detection['confidence'],
                    frame_number=detection['frame_number'],
                    bounding_box=bbox
                )
                detection_objects.append(detection_obj)
        # Bulk create detection results
        DetectionResult.objects.bulk_create(detection_objects)
        # Create summary video if detections found
        summary_video_path = None
        if detections:
            timestamps = [d['timestamp'] for d in detections]
            # Generate temporary file for summary video
            temp_dir = tempfile.mkdtemp()
            video_name = f"video_{video.id}"
            summary_filename = f"{video_name}_summary.mp4"
            temp_summary_path = os.path.join(temp_dir, summary_filename)
            
            # Create summary video
            success = video_processor.create_summary_video(
                video_path, 
                timestamps, 
                temp_summary_path
            )
            if success:
                # Create ProcessedVideo record and save to database
                processed_video = ProcessedVideo(
                    case=case,
                    original_video=video,
                    total_detections=len(detections),
                    summary_duration=len(timestamps) * (video_processor.buffer_before + video_processor.buffer_after)
                )
                processed_video.save_processed_from_file(temp_summary_path, summary_filename)
                processed_video.save()
                summary_video_path = f"/cases/video/{processed_video.id}/"
                
                # Clean up temporary file
                try:
                    os.remove(temp_summary_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temporary file {temp_summary_path}: {e}")
        # Update completion status
        video.processed = True
        video.processing_completed_at = timezone.now()
        video.save()
        case.status = 'completed'
        case.save()
        result = {
            'status': 'completed',
            'video_id': video_id,
            'total_detections': len(detections),
            'suspects_found': len(set(d['suspect_index'] for d in detections)),
            'summary_video_url': summary_video_path,
            'processing_time': (timezone.now() - video.processing_started_at).total_seconds()
        }
        return result
    except Exception as exc:
        # Handle task failure
        try:
            video = VideoUpload.objects.get(id=video_id)
            case = video.case
            case.status = 'error'
            case.save()
        except:
            pass
        
        # Log the error
        error_msg = f"Task failed: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Retry the task
        # Clean up temporary video file(s)
        try:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60, exc=exc)
        
        return {
            'status': 'error',
            'error': str(exc),
            'video_id': video_id
        }
    finally:
        # Clean up temporary video file
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)



@shared_task
def process_suspect_image_task(suspect_id: int):
    try:
        suspect = SuspectImage.objects.get(id=suspect_id)
        print('DeepFaceClient doing...')
        face_client = DeepFaceClient()
        print('DeepFaceClient done.')

        print('extract_features...')
        
        # Create temporary file from database-stored image
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, f"temp_suspect_{suspect.id}.{suspect.image_type}")
        
        # Write image data to temporary file
        with open(temp_image_path, 'wb') as f:
            f.write(suspect.get_image_data())
        
        try:
            detected_faces = face_client.detect_faces_in_image(temp_image_path)
            print(f'Detected {len(detected_faces)} faces in suspect image')

            if detected_faces:
                image = cv2.imread(temp_image_path)
                for i, (bbox, conf) in enumerate(detected_faces):
                    x, y, w, h = bbox
                    print(f'Face {i+1}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}')
                    face_crop = image[y:y+h+20, x:x+w+20]
        

            if detected_faces:
                largest_face = max(detected_faces, key=lambda x: x[0][2] * x[0][3])
                x, y, w, h = largest_face[0]
                image = cv2.imread(temp_image_path)

                temp_face_dir = 'C:\\Users\\asus\\OneDrive\\Pictures\\CE\\SEM-4\\Projects\\Detectify\\temp'
                os.makedirs(temp_face_dir, exist_ok=True)
                temp_face_path = os.path.join(temp_face_dir, f"temp_cropped_face_{suspect.id}.{suspect.image_type}")

                h_img, w_img = image.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                face_crop = image[y1:y2, x1:x2]

                print(f"Cropped face shape: {face_crop.shape}")

                # Try saving with OpenCV first
                success = cv2.imwrite(temp_face_path, face_crop)
                if not success:
                    print("cv2.imwrite failed, trying Pillow save...")

                    # Convert BGR to RGB for Pillow
                    try:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(face_crop_rgb)
                        pil_img.save(temp_face_path, format='JPEG')
                        print(f"Image saved with Pillow at {temp_face_path}")
                    except Exception as e:
                        error_msg = f"Failed to save image with Pillow: {e}"
                        print(error_msg)
                        return {
                            'status': 'error',
                            'error': error_msg,
                            'suspect_id': suspect_id
                        }
                else:
                    print(f"Image saved with OpenCV at {temp_face_path}")

                encoding = face_client.deepface.represent(
                    img_path=temp_face_path,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                features = np.array(encoding[0]['embedding'])
                if len(features) != face_client.output_shape:
                    if len(features) < face_client.output_shape:
                        features = np.pad(features, (0, face_client.output_shape - len(features)))
                    else:
                        features = features[:face_client.output_shape]

                print('extract_features done.')
                if features.sum() == 0:
                    print(f'Warning: Zero encoding extracted for suspect {suspect.id}')
                else:
                    print(f'Valid encoding extracted with magnitude: {np.linalg.norm(features):.3f}')
                suspect.face_encoding = features.tolist()
                suspect.processed = True
                suspect.save()

                # Clean up temporary files
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                return {
                    'status': 'completed',
                    'suspect_id': suspect_id,
                    'encoding_shape': features.shape
                }
            else:
                print(f'No faces detected in suspect image, skipping encoding.')
                return {
                    'status': 'error',
                    'error': 'No face detected in suspect image',
                    'suspect_id': suspect_id
                }
        finally:
            # Clean up temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    except Exception as exc:
        error_msg = f"Suspect processing failed: {str(exc)}"
        print(error_msg)

        return {
            'status': 'error',
            'error': str(exc),
            'suspect_id': suspect_id
        }



@shared_task
def cleanup_old_files_task():
    """
    Cleanup old temporary files and processed videos
    """
    try:
        import time
        from django.conf import settings
        
        # Define cleanup paths
        temp_dirs = [
            os.path.join(settings.MEDIA_ROOT, 'temp'),
            '/tmp'
        ]
        
        current_time = time.time()
        cleanup_age = 24 * 60 * 60  # 24 hours
        
        cleaned_files = 0
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    if filename.startswith('temp_face_'):
                        file_path = os.path.join(temp_dir, filename)
                        file_age = current_time - os.path.getctime(file_path)
                        
                        if file_age > cleanup_age:
                            try:
                                os.remove(file_path)
                                cleaned_files += 1
                            except:
                                pass
        
        return {
            'status': 'completed',
            'cleaned_files': cleaned_files
        }
        
    except Exception as exc:
        return {
            'status': 'error',
            'error': str(exc)
        }


@shared_task
def extract_video_metadata_task(video_id: int):
    """
    Extract and save video metadata
    
    Args:
        video_id (int): ID of VideoUpload instance
    """
    try:
        video = VideoUpload.objects.get(id=video_id)
        video_processor = VideoProcessor()
        
        metadata = video_processor.extract_video_metadata(video.video_file.path)
        
        # Update video record with metadata
        video.duration = metadata.get('duration', 0)
        video.fps = int(metadata.get('fps', 0))
        video.save()
        
        return {
            'status': 'completed',
            'video_id': video_id,
            'metadata': metadata
        }
        
    except Exception as exc:
        return {
            'status': 'error',
            'error': str(exc),
            'video_id': video_id
        }

