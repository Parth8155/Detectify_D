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
from typing import Generator, Optional, Tuple, Any

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
    
    def __init__(self, video_path: str, detection_model_path: str = "face_detection_yunet_2023mar.onnx"):
        """Initialize the video processor.
        
        Args:
            video_path: Path to the video file
            detection_model_path: Path to the face detection model
        """
        self.video_path = video_path
        self.face_detector = FaceDetector(detection_model_path)
        self.cap = None
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer for frames
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'fps': 0,
            'processing_time': 0
        }
        # Internal: effective total frames after skipping and current skip factor
        self._effective_total_frames = 0
        self._frame_skip = 5
        # Run a full-resolution detection every N processed frames for accuracy
        # Set to 1 to always run full-res (slow). Default 5 means full detection every 5th processed frame.
        self._full_res_every = 5
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
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process a single frame with face detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, number_of_faces_detected)
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
        
        # Draw rectangles around detected faces
        faces_count = 0
        for face in faces:
            # Extract bounding box coordinates
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optional: Add confidence score
            confidence = face[14] if len(face) > 14 else 0.0
            cv2.putText(frame, f'{confidence:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            faces_count += 1
        
        # Update statistics
        processing_time = time.time() - start_time
        with self._lock:
            self.stats['processed_frames'] += 1
            self.stats['faces_detected'] += faces_count
            self.stats['processing_time'] = processing_time
        
        return frame, faces_count
    
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
    
    def generate_mjpeg_stream(self, target_fps: int = 30, quality: int = 80) -> Generator[bytes, None, None]:
        """Generate MJPEG stream for HTTP streaming.
        
        Args:
            target_fps: Target frames per second
            quality: JPEG quality (1-100)
            
        Yields:
            MJPEG frame bytes
        """
        if not self._initialize_video_capture():
            return
        
        self.is_streaming = True
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()

        # Compute skip factor: number of source frames to skip between processed frames
        try:
            if self.fps and target_fps > 0:
                self._frame_skip = max(1, int(round(self.fps / float(target_fps))))
            else:
                self._frame_skip = 1
        except Exception:
            self._frame_skip = 1

        # Effective total frames after skipping (used for progress)
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
                processed_frame, faces_count = self._process_frame(frame)
                
                # Encode frame as JPEG
                jpeg_bytes = self._encode_frame_jpeg(processed_frame, quality)
                
                # Create MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                
                # Skip frames efficiently to reduce processing load
                if self._frame_skip > 1:
                    # we've already read one frame (the processed one), grab the next (skip-1) frames
                    for _ in range(self._frame_skip - 1):
                        # grab() moves to next frame without decoding (faster than read)
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
            logger.error(f"MJPEG streaming error: {e}")
        finally:
            self.cleanup()
    
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
                processed_frame, faces_count = self._process_frame(frame)
                
                # Encode frame as base64
                frame_b64 = self._encode_frame_base64(processed_frame, quality)
                
                # Create frame data with metadata
                frame_data = {
                    'type': 'frame',
                    'data': frame_b64,
                    'timestamp': time.time(),
                    'frame_number': self.stats['processed_frames'],
                    'faces_detected': faces_count,
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
            else:
                stats['progress'] = 0
                stats['avg_faces_per_frame'] = 0
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
                      detection_model_path: str = "face_detection_yunet_2023mar.onnx") -> VideoStreamProcessor:
        """Create a new streaming session.
        
        Args:
            session_id: Unique session identifier
            video_path: Path to the video file
            detection_model_path: Path to the face detection model
            
        Returns:
            VideoStreamProcessor instance
        """
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].cleanup()
            
            processor = VideoStreamProcessor(video_path, detection_model_path)
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
