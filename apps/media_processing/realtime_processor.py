"""
Real-time video processing with face detection and streaming capabilities.
Supports both MJPEG streaming and WebSocket streaming.
"""

import cv2
import numpy as np
import base64
import threading
import time
import queue
import logging
from io import BytesIO
from PIL import Image
import json
import tempfile
import os
from typing import Generator, Optional, Tuple, Any, List, Dict
from django.utils import timezone
from channels.db import database_sync_to_async
from .deepface_client import DeepFaceClient
from ..cases.models import DetectionResult, ProcessedVideo

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV's YuNet model."""
    
    def __init__(self, model_path: str = "face_detection_yunet_2023mar.onnx"):
        """Initialize the face detector.
        
        Args:
            model_path: Path to the YuNet ONNX model file
        """
        self.model_path = model_path
        self.detector = None
        # Default detector input size (smaller -> faster)
        self.input_size = (320, 240)
        # Initialize detector instance
        try:
            # Delay heavy initialization until possible; keep call here for compatibility
            self._initialize_detector()
        except Exception:
            # If initialization fails here, set detector to None and continue
            self.detector = None
    
    def _initialize_detector(self):
        """Initialize the YuNet face detector."""
        try:
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 240),  # Default size, will be updated
                0.9,  # Score threshold
                0.3,  # NMS threshold
                5000  # Top K
            )
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.detector = None
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in the given frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of face detections with bounding boxes
        """
        if self.detector is None:
            return []
        
        try:
            # Update input size
            height, width = frame.shape[:2]
            self.detector.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.detector.detect(frame)
            
            if faces is not None:
                return faces.tolist()
            return []
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def detect_faces_resized(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> list:
        """Detect faces on a resized copy of the frame and map results back.

        Args:
            frame: Original frame
            target_size: (width, height) to resize detector input to

        Returns:
            List of detections scaled to original frame coordinates
        """
        if self.detector is None:
            return []

        try:
            target_size = target_size or self.input_size
            det_w, det_h = int(target_size[0]), int(target_size[1])
            orig_h, orig_w = frame.shape[:2]

            # Resize frame for faster detection
            small = cv2.resize(frame, (det_w, det_h))

            # Set detector input size and run detection on small frame
            self.detector.setInputSize((det_w, det_h))
            _, faces = self.detector.detect(small)

            if faces is None:
                return []

            # Scale detections back to original frame size
            sx = orig_w / float(det_w)
            sy = orig_h / float(det_h)

            scaled = []
            for f in faces.tolist():
                # f is typically length 15: x,y,w,h,score, lmkx1, lmky1, ...
                f = list(f)
                # Scale x, y, w, h
                f[0] = f[0] * sx
                f[1] = f[1] * sy
                f[2] = f[2] * sx
                f[3] = f[3] * sy

                # Scale landmarks if present (indices 5..14)
                for i in range(5, min(len(f), 15)):
                    if (i - 5) % 2 == 0:
                        # x coordinate
                        f[i] = f[i] * sx
                    else:
                        # y coordinate
                        f[i] = f[i] * sy

                scaled.append(f)

            return scaled

        except Exception as e:
            logger.error(f"Resized face detection error: {e}")
            return []


class VideoStreamProcessor:
    """Real-time video processor with face detection and streaming."""
    
    def __init__(self, video_path: str, detection_model_path: str = "face_detection_yunet_2023mar.onnx", 
                 video_obj=None, suspect_encodings: List[np.ndarray] = None, suspect_mapping: Dict = None):
        """Initialize the video processor.
        
        Args:
            video_path: Path to the video file
            detection_model_path: Path to the face detection model
            video_obj: VideoUpload model instance for saving DetectionResult
            suspect_encodings: List of suspect face encodings for recognition
            suspect_mapping: Mapping from encoding index to suspect object
        """
        self.video_path = video_path
        self.face_detector = FaceDetector(detection_model_path)
        self.cap = None
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer for frames
        
        # Recognition and database saving
        self.video_obj = video_obj
        self.suspect_encodings = suspect_encodings or []
        self.suspect_mapping = suspect_mapping or {}
        self.confidence_threshold = 0.85
        
        # Store detections to save in batch (avoid async/sync issues)
        self.pending_detections = []
        self.detection_lock = threading.Lock()
        
        # Store detection timestamps for summary video creation
        self.detection_timestamps = []
        
        # Import here to avoid circular imports
        try:
            self.face_client = DeepFaceClient() if suspect_encodings else None
            self.DetectionResult = DetectionResult
            self.ProcessedVideo = ProcessedVideo
        except ImportError:
            self.face_client = None
            self.DetectionResult = None
            self.ProcessedVideo = None
        
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'suspects_detected': 0,  # New stat for suspect detections
            'fps': 0,
            'processing_time': 0
        }
        # Internal: effective total frames after skipping and current skip factor
        self._effective_total_frames = 0
        self._frame_skip = 5
        # Run a full-resolution detection every N processed frames for accuracy
        # Set to 1 to always run full-res (slow). Default 5 means full detection every 5th processed frame.
        self._full_res_every = 5
        # Save detections to DB every N frames to avoid losing data
        self._save_interval = 50  # Save every 50 processed frames
        self._lock = threading.Lock()
    
    def _initialize_video_capture(self) -> bool:
        """Initialize video capture."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {self.video_path}")
                return False
            
            # Get video properties
            self.stats['total_frames'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
            self.stats['fps'] = self.fps
            
            logger.info(f"Video initialized: {self.stats['total_frames']} frames at {self.fps} FPS")
            return True
        except Exception as e:
            logger.error(f"Video initialization error: {e}")
            return False
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Process a single frame with face detection and recognition.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, number_of_faces_detected, suspects_detected)
        """
        start_time = time.time()
        
        # Hybrid detection: use resized detection for speed, but occasionally
        # run full-resolution detection for improved accuracy.
        try:
            use_full = False
            try:
                # stats['processed_frames'] is the count of processed frames so far
                use_full = (self._full_res_every and (self.stats['processed_frames'] % self._full_res_every == 0))
            except Exception:
                use_full = False

            if use_full:
                faces = self.face_detector.detect_faces(frame)
            else:
                faces = self.face_detector.detect_faces_resized(frame, target_size=(320, 240))
        except Exception:
            faces = self.face_detector.detect_faces(frame)
        
        # Draw rectangles around detected faces and perform recognition
        faces_count = 0
        suspects_detected = 0
        
        # Calculate current timestamp for this frame
        if self.fps > 0:
            current_timestamp = self.stats['processed_frames'] / (self.fps / self._frame_skip)
        else:
            current_timestamp = time.time()
        
        for face in faces:
            # Extract bounding box coordinates
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            
            # Skip very small faces (likely false positives)
            if w < 40 or h < 40:
                continue
            
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optional: Add confidence score
            confidence = face[14] if len(face) > 14 else 0.0
            cv2.putText(frame, f'{confidence:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            faces_count += 1
            
            # Perform suspect recognition if we have suspect encodings
            if self.suspect_encodings and self.face_client and self.video_obj:
                try:
                    suspect_detected = self._recognize_suspect_in_face(frame, x, y, w, h, current_timestamp)
                    if suspect_detected:
                        suspects_detected += 1
                        # Draw red rectangle for recognized suspects
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(frame, 'SUSPECT', (x, y - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    logger.error(f"Error in suspect recognition: {e}")
        
        # Update statistics
        processing_time = time.time() - start_time
        with self._lock:
            self.stats['processed_frames'] += 1
            self.stats['faces_detected'] += faces_count
            self.stats['suspects_detected'] += suspects_detected
            self.stats['processing_time'] = processing_time
            
            # Periodically save pending detections to avoid data loss
            if (self.stats['processed_frames'] % self._save_interval == 0 and 
                self.pending_detections and self.DetectionResult):
                try:
                    # Save in a separate thread to avoid blocking
                    import threading
                    save_thread = threading.Thread(target=self.save_pending_detections)
                    save_thread.daemon = True
                    save_thread.start()
                except Exception as e:
                    logger.error(f"Error starting periodic save thread: {e}")
        
        return frame, faces_count, suspects_detected
    
    def _recognize_suspect_in_face(self, frame: np.ndarray, x: int, y: int, w: int, h: int, timestamp: float) -> bool:
        """Recognize if a detected face matches any suspect.
        
        Args:
            frame: Full frame image
            x, y, w, h: Bounding box coordinates
            timestamp: Current timestamp in video
            
        Returns:
            True if suspect is recognized and saved to database
        """
        try:
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return False
            
            # Save face to temporary file for recognition
            temp_dir = tempfile.gettempdir()
            temp_face_path = os.path.join(temp_dir, f"temp_face_stream_{timestamp}_{x}_{y}.jpg")
            cv2.imwrite(temp_face_path, face_region)
            
            try:
                # Extract features from detected face
                face_encoding = self.face_client.extract_features(temp_face_path)
                
                # Compare with all suspect encodings
                for i, suspect_encoding in enumerate(self.suspect_encodings):
                    similarity = self.face_client.compare_faces(face_encoding, suspect_encoding)
                    
                    if similarity >= self.confidence_threshold:
                        # Found a match! Store detection data for batch saving
                        suspect = self.suspect_mapping.get(i)
                        if suspect and self.DetectionResult:
                            detection_data = {
                                'video': self.video_obj,
                                'suspect': suspect,
                                'timestamp': timestamp,
                                'confidence': similarity,
                                'frame_number': self.stats['processed_frames'],
                                'bounding_box': (x, y, w, h)
                            }
                            
                            # Add to pending detections (thread-safe)
                            with self.detection_lock:
                                self.pending_detections.append(detection_data)
                                # Also store timestamp for summary video
                                self.detection_timestamps.append(timestamp)
                            
                            logger.info(f"Queued detection: suspect {suspect.id} at {timestamp:.2f}s with confidence {similarity:.3f}")
                            return True
                
                return False
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                    
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return False
    
    def save_pending_detections(self):
        """Save all pending detections to database in batch."""
        if not self.pending_detections or not self.DetectionResult:
            return 0
            
        try:
            with self.detection_lock:
                if not self.pending_detections:
                    return 0
                    
                # Create DetectionResult objects
                detection_objects = []
                for detection_data in self.pending_detections:
                    detection_obj = self.DetectionResult(**detection_data)
                    detection_objects.append(detection_obj)
                
                # Bulk create all detections
                created_count = len(self.DetectionResult.objects.bulk_create(detection_objects))
                logger.info(f"Saved {created_count} detection results to database")
                
                # Clear pending detections
                self.pending_detections.clear()
                return created_count
                
        except Exception as e:
            logger.error(f"Error saving pending detections: {e}")
            return 0
    
    def create_summary_video(self):
        """Create summary video from detection timestamps."""
        if not self.detection_timestamps or not self.ProcessedVideo:
            logger.info("No detections found or ProcessedVideo model not available")
            return None
            
        try:
            # Import video processor for summary creation
            from .video_processor import VideoProcessor
            import tempfile
            from django.conf import settings
            
            video_processor = VideoProcessor()
            
            # Get unique timestamps (remove duplicates and sort)
            unique_timestamps = sorted(list(set(self.detection_timestamps)))
            logger.info(f"Creating summary video with {len(unique_timestamps)} detection timestamps")
            
            # Generate temporary file for summary video
            temp_dir = tempfile.mkdtemp()
            case = self.video_obj.case
            video_name = f"video_{self.video_obj.id}"
            summary_filename = f"{video_name}_summary.mp4"
            temp_summary_path = os.path.join(temp_dir, summary_filename)
            
            # Create summary video using video_processor
            success = video_processor.create_summary_video(
                self.video_path,
                unique_timestamps,
                temp_summary_path
            )
            
            if success:
                logger.info(f"Summary video created successfully at {temp_summary_path}")
                
                # Create ProcessedVideo record and save to database
                processed_video = self.ProcessedVideo(
                    case=case,
                    original_video=self.video_obj,
                    total_detections=len(unique_timestamps),
                    summary_duration=len(unique_timestamps) * (video_processor.buffer_before + video_processor.buffer_after)
                )
                
                # Save processed video data to database
                processed_video.save_processed_from_file(temp_summary_path, summary_filename)
                processed_video.save()
                
                logger.info(f"ProcessedVideo record created with ID {processed_video.id}")
                
                # Clean up temporary file
                try:
                    os.remove(temp_summary_path)
                    os.rmdir(temp_dir)
                    logger.info(f"Cleaned up temporary file {temp_summary_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file {temp_summary_path}: {e}")
                    
                return processed_video
            else:
                logger.error("Failed to create summary video")
                # Clean up temporary directory even if video creation failed
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    pass
                return None
                
        except Exception as e:
            logger.error(f"Error creating summary video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _encode_frame_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """Encode frame as JPEG bytes.
        
        Args:
            frame: Input frame
            quality: JPEG quality (1-100)
            
        Returns:
            JPEG encoded bytes
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()
    
    def _encode_frame_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """Encode frame as base64 string.
        
        Args:
            frame: Input frame
            quality: JPEG quality (1-100)
            
        Returns:
            Base64 encoded JPEG string
        """
        jpeg_bytes = self._encode_frame_jpeg(frame, quality)
        return base64.b64encode(jpeg_bytes).decode('utf-8')
   
    def generate_websocket_frames(self, target_fps: int = 30, quality: int = 80) -> Generator[dict, None, None]:
        """Generate frames for WebSocket streaming.
        
        Args:
            target_fps: Target frames per second
            quality: JPEG quality (1-100)
            
        Yields:
            Dictionary with frame data and metadata
        """
        if not self._initialize_video_capture():
            return
        
        self.is_streaming = True
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()

        # Compute skip factor
        try:
            if self.fps and target_fps > 0:
                self._frame_skip = max(1, int(round(self.fps / float(target_fps))))
            else:
                self._frame_skip = 1
        except Exception:
            self._frame_skip = 1

        # Effective total frames after skipping
        if self.stats['total_frames'] > 0 and self._frame_skip > 1:
            self._effective_total_frames = max(1, int(self.stats['total_frames'] / self._frame_skip))
        else:
            self._effective_total_frames = self.stats['total_frames']
        
        try:
            while self.is_streaming and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Video processing completed")
                    break
                
                # Process frame with face detection
                processed_frame, faces_count, suspects_count = self._process_frame(frame)
                
                # Encode frame as base64
                frame_b64 = self._encode_frame_base64(processed_frame, quality)
                
                # Create frame data with metadata
                frame_data = {
                    'type': 'frame',
                    'data': frame_b64,
                    'timestamp': time.time(),
                    'frame_number': self.stats['processed_frames'],
                    'faces_detected': faces_count,
                    'suspects_detected': suspects_count,  # Add suspect count
                    'stats': self.get_stats()
                }
                
                yield frame_data

                # Skip frames efficiently
                if self._frame_skip > 1:
                    for _ in range(self._frame_skip - 1):
                        try:
                            self.cap.grab()
                        except Exception:
                            break

                # Control frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()
                
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
        finally:
            # Save any pending detections before cleanup
            if hasattr(self, 'save_pending_detections'):
                try:
                    self.save_pending_detections()
                except Exception as e:
                    logger.error(f"Error saving pending detections during cleanup: {e}")
            
            # Create summary video after processing is complete
            if hasattr(self, 'video_obj') and self.video_obj:
                try:
                    logger.info("Creating summary video...")
                    # Use threading to avoid async context issues
                    import threading
                    
                    def create_summary_and_complete():
                        try:
                            processed_video = self.create_summary_video()
                            if processed_video:
                                logger.info(f"Summary video creation successful")
                            
                            # Mark video processing as complete
                            self.video_obj.status = 'completed'
                            self.video_obj.save()
                            
                            logger.info(f"Video processing completed. Total detections: {len(getattr(self, 'detection_timestamps', []))}")
                        except Exception as e:
                            logger.error(f"Error in summary video creation thread: {e}")
                    
                    # Run summary creation in separate thread to avoid async context issues
                    summary_thread = threading.Thread(target=create_summary_and_complete)
                    summary_thread.start()
                    summary_thread.join()  # Wait for completion
                    
                except Exception as e:
                    logger.error(f"Error creating summary video during cleanup: {e}")
            
            self.cleanup()
    
    def get_stats(self) -> dict:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        with self._lock:
            stats = self.stats.copy()
            total_ref = self._effective_total_frames or stats['total_frames']
            if stats['processed_frames'] > 0 and total_ref > 0:
                stats['progress'] = (stats['processed_frames'] / total_ref) * 100
                stats['avg_faces_per_frame'] = stats['faces_detected'] / stats['processed_frames']
                stats['avg_suspects_per_frame'] = stats['suspects_detected'] / stats['processed_frames']
            else:
                stats['progress'] = 0
                stats['avg_faces_per_frame'] = 0
                stats['avg_suspects_per_frame'] = 0
            
            # Add pending detection count
            with self.detection_lock:
                stats['pending_detections'] = len(self.pending_detections)
            
            return stats
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_streaming = False
    
    def cleanup(self):
        """Clean up resources."""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None


class StreamingSession:
    """Manages multiple streaming sessions."""
    
    def __init__(self):
        self.active_sessions = {}
        self._lock = threading.Lock()
    
    def create_session(self, session_id: str, video_path: str, 
                      detection_model_path: str = "face_detection_yunet_2023mar.onnx",
                      video_obj=None, suspect_encodings: List[np.ndarray] = None, 
                      suspect_mapping: Dict = None) -> VideoStreamProcessor:
        """Create a new streaming session.
        
        Args:
            session_id: Unique session identifier
            video_path: Path to the video file
            detection_model_path: Path to the face detection model
            video_obj: VideoUpload model instance for saving DetectionResult
            suspect_encodings: List of suspect face encodings for recognition
            suspect_mapping: Mapping from encoding index to suspect object
            
        Returns:
            VideoStreamProcessor instance
        """
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].cleanup()
            
            processor = VideoStreamProcessor(
                video_path, 
                detection_model_path,
                video_obj=video_obj,
                suspect_encodings=suspect_encodings,
                suspect_mapping=suspect_mapping
            )
            self.active_sessions[session_id] = processor
            return processor
    
    def get_session(self, session_id: str) -> Optional[VideoStreamProcessor]:
        """Get an existing session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            VideoStreamProcessor instance or None
        """
        with self._lock:
            return self.active_sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove and cleanup a session.
        
        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].cleanup()
                del self.active_sessions[session_id]
    
    def cleanup_all(self):
        """Cleanup all active sessions."""
        with self._lock:
            for processor in self.active_sessions.values():
                processor.cleanup()
            self.active_sessions.clear()


# Global streaming session manager
streaming_sessions = StreamingSession()
