"""
Celery tasks for asynchronous video processing in Detectify
Supports normal batch processing (WebSocket code removed from this module)
"""

from celery import shared_task
from django.core.files.base import ContentFile
from django.utils import timezone
from typing import List, Dict, Callable, Optional, Tuple, Any
import os
import tempfile
import json
import traceback
import numpy as np
import cv2
import base64
import time

from apps.cases.models import VideoUpload, SuspectImage, DetectionResult, ProcessedVideo, Case
from .video_processor import VideoProcessor
from .deepface_client_optimized import DeepFaceClient  # Use optimized singleton version
from django.conf import settings
from PIL import Image
from apps.cases.models import Case

# Global instances to avoid repeated initialization (expensive operations)
_unified_processor_instance = None
_deepface_client_instance = None

def get_unified_processor():
    """Get or create a singleton UnifiedVideoProcessor instance"""
    global _unified_processor_instance
    if _unified_processor_instance is None:
        _unified_processor_instance = UnifiedVideoProcessor()
    return _unified_processor_instance

def get_deepface_client():
    """Get or create a singleton DeepFaceClient instance"""
    global _deepface_client_instance
    if _deepface_client_instance is None:
        _deepface_client_instance = DeepFaceClient()
    return _deepface_client_instance

class UnifiedVideoProcessor:
    """Video processor for normal (batch) processing only.

    WebSocket-specific branches were removed to simplify this module.
    """

    def __init__(self):
        self._face_client = None
        self.confidence_threshold = getattr(settings, 'FACE_RECOGNITION_CONFIDENCE_THRESHOLD', 0.85)
        self.frame_skip = getattr(settings, 'VIDEO_FRAME_SKIP', 5)

    @property
    def face_client(self):
        if self._face_client is None:
            self._face_client = get_deepface_client()
        return self._face_client

    def process_video_unified(self,
                             video_path: str,
                             suspect_encodings: List[np.ndarray],
                             suspect_mapping: Dict[int, SuspectImage],
                             video_obj: VideoUpload,
                             mode: str = 'normal',
                             progress_callback: Optional[Callable] = None,
                             frame_callback: Optional[Callable] = None) -> Tuple[List[Dict], Optional[bytes]]:
        """
        Process a video in normal (batch) mode. WebSocket-specific behavior removed.
        Returns a list of detection dicts and None for the summary bytes (handled separately).
        """
        detections: List[Dict] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_number = 0
        detections_found = 0
        processed_frames = 0
        start_time = time.time()

        try:
            min_face_size = 40

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps if fps > 0 else 0

                process_frame = (frame_number % self.frame_skip == 0)

                if process_frame:
                    processed_frames += 1

                    frame_resized = frame
                    detected_faces = self.face_client.detect_faces_in_frame(frame_resized)
                    faces_count = len(detected_faces)
                    suspects_detected = 0

                    frame_with_detections = frame.copy()

                    for bounding_box, detection_confidence in detected_faces:
                        x, y, w, h = bounding_box
                        if w < min_face_size or h < min_face_size:
                            continue

                        cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        face_region = frame[y:y+h, x:x+w]
                        if face_region.size == 0:
                            continue

                        temp_dir = tempfile.gettempdir()
                        temp_face_path = os.path.join(temp_dir, f"temp_face_{frame_number}_{x}_{y}.jpg")
                        cv2.imwrite(temp_face_path, face_region)

                        try:
                            face_encoding = self.face_client.extract_features(temp_face_path)

                            for i, suspect_encoding in enumerate(suspect_encodings):
                                similarity = self.face_client.compare_faces(face_encoding, suspect_encoding)

                                if similarity >= self.confidence_threshold:
                                    cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                    cv2.putText(frame_with_detections, f'SUSPECT: {similarity:.2f}',
                                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                                    suspects_detected += 1

                                    detection = {
                                        'timestamp': timestamp,
                                        'confidence': similarity,
                                        'bounding_box': (x, y, w, h),
                                        'suspect_index': i,
                                        'frame_number': frame_number
                                    }
                                    detections.append(detection)
                                    detections_found += 1

                                    break

                        finally:
                            if os.path.exists(temp_face_path):
                                os.remove(temp_face_path)

                    if frame_callback:
                        try:
                            _, buffer = cv2.imencode('.jpg', frame_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            frame_data = {
                                'type': 'frame',
                                'data': frame_b64,
                                'timestamp': timestamp,
                                'frame_number': frame_number,
                                'faces_detected': faces_count,
                                'suspects_detected': suspects_detected,
                                'stats': {
                                    'processed_frames': processed_frames,
                                    'total_frames': total_frames,
                                    'total_detections': detections_found,
                                    'fps': fps
                                }
                            }
                            try:
                                frame_callback(frame_data)
                            except Exception:
                                pass
                        except Exception:
                            pass

                if progress_callback:
                    progress_callback(frame_number + 1, timestamp, detections_found)

                frame_number += 1

        finally:
            cap.release()

        return detections, None

    def create_summary_video_from_detections(self,
                                           video_path: str,
                                           detections: List[Dict],
                                           output_path: str) -> bool:
        """Create summary video from detection results"""
        if not detections:
            return False

        try:
            video_processor = VideoProcessor()
            timestamps = [d['timestamp'] for d in detections]
            return video_processor.create_summary_video(video_path, timestamps, output_path)
        except Exception as e:
            print(f"Error creating summary video: {str(e)}")
            return False

    def create_summary_video_from_detections(self, 
                                           video_path: str, 
                                           detections: List[Dict], 
                                           output_path: str) -> bool:
        """Create summary video from detection results"""
        print("creating summary video")
        if not detections:
            return False
        
        try:
            video_processor = VideoProcessor()
            timestamps = [d['timestamp'] for d in detections]
            return video_processor.create_summary_video(video_path, timestamps, output_path)
        except Exception as e:
            print(f"Error creating summary video: {str(e)}")
            return False

@shared_task(bind=True, max_retries=3)
def process_video_task(self, video_id: int, suspect_ids: List[int]):
    """
    Process video to detect suspect faces using unified processor
    
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
        
        # Initialize unified processor (use singleton to avoid repeated model loading)
        processor = get_unified_processor()
        
        # Extract suspect encodings
        suspect_encodings = []
        suspect_mapping = {}
        
        for i, suspect in enumerate(suspects):
            if suspect.face_encoding:
                encoding = json.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                suspect_encodings.append(np.array(encoding))
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
                        encoding = processor.face_client.extract_features(temp_image_path)
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
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
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
        
        # Process video using unified processor in normal mode
        detections, _ = processor.process_video_unified(
            video_path=video_path,
            suspect_encodings=suspect_encodings,
            suspect_mapping=suspect_mapping,
            video_obj=video,
            mode='normal',
            progress_callback=progress_callback
        )
        
        # Save detection results
        detection_objects = []
        for detection in detections:
            suspect = suspect_mapping.get(detection['suspect_index'])
            if suspect:
                detection_obj = DetectionResult(
                    video=video,
                    suspect=suspect,
                    timestamp=detection['timestamp'],
                    confidence=detection['confidence'],
                    frame_number=detection['frame_number'],
                    bounding_box=detection['bounding_box']
                )
                detection_objects.append(detection_obj)
        
        # Bulk create detection results
        DetectionResult.objects.bulk_create(detection_objects)
        
        # Create summary video if detections found
        summary_video_path = None
        if detections:
            # Generate temporary file for summary video
            temp_dir_summary = tempfile.mkdtemp()
            video_name = f"video_{video.id}"
            summary_filename = f"{video_name}_summary.mp4"
            temp_summary_path = os.path.join(temp_dir_summary, summary_filename)
            
            # Create summary video using unified processor
            success = processor.create_summary_video_from_detections(
                video_path, detections, temp_summary_path
            )
            
            if success:
                # Create ProcessedVideo record and save to database
                processed_video = ProcessedVideo(
                    case=case,
                    original_video=video,
                    total_detections=len(detections),
                    summary_duration=len(detections) * 5  # Estimate duration
                )
                processed_video.save_processed_from_file(temp_summary_path, summary_filename)
                processed_video.save()
                summary_video_path = f"/cases/video/{processed_video.id}/"
                
                # Clean up temporary file
                try:
                    os.remove(temp_summary_path)
                    os.rmdir(temp_dir_summary)
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





def get_suspect_data_for_processing(case_id: int) -> Tuple[List[np.ndarray], Dict[int, SuspectImage]]:
    """
    Helper function to get suspect encodings and mapping for a case
    
    Args:
        case_id (int): Case ID
        
    Returns:
        Tuple of (suspect_encodings, suspect_mapping)
    """    
    try:
        case = Case.objects.get(id=case_id)
        suspects = case.suspect_images.filter(processed=True)  # Fixed: use suspect_images, not suspects
        
        suspect_encodings = []
        suspect_mapping = {}
        processor = get_unified_processor()  # Use singleton to avoid repeated model loading
        
        for i, suspect in enumerate(suspects):
            if suspect.face_encoding:
                encoding = json.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                suspect_encodings.append(np.array(encoding))
                suspect_mapping[i] = suspect
            elif suspect.image_data:
                # Generate encoding if not exists
                try:
                    temp_dir = tempfile.gettempdir()
                    temp_image_path = os.path.join(temp_dir, f"temp_suspect_{suspect.id}.{suspect.image_type}")
                    
                    with open(temp_image_path, 'wb') as f:
                        f.write(suspect.get_image_data())
                    
                    try:
                        encoding = processor.face_client.extract_features(temp_image_path)
                        suspect.face_encoding = encoding.tolist()
                        suspect.processed = True
                        suspect.save()
                        
                        suspect_encodings.append(encoding)
                        suspect_mapping[i] = suspect
                    finally:
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                except Exception as e:
                    print(f"Error processing suspect {suspect.id}: {str(e)}")
        
        return suspect_encodings, suspect_mapping
        
    except Exception as e:
        print(f"Error getting suspect data: {str(e)}")
        return [], {}






@shared_task
def process_suspect_image_task(suspect_id: int):
    try:
        suspect = SuspectImage.objects.get(id=suspect_id)
        print('Getting DeepFaceClient instance...')
        face_client = get_deepface_client()  # Use singleton to avoid repeated model loading
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
                print("VGG-Face for suspect")
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

