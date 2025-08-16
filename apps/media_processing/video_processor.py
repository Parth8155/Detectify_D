"""
Video processing utilities for Detectify
Handles video analysis, frame extraction, and summary video creation
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
from django.conf import settings
from .deepface_client import DeepFaceClient
import tempfile
import time

class VideoProcessor:

    def analyze_video_for_unique_faces(self, video_path: str, similarity_threshold: float = 0.7, progress_callback=None):
        """
        Stub: Unique face extraction disabled.
        """
        return []
    """Handles video processing operations for face detection"""
    
    def __init__(self):
        self.face_recognition_client = DeepFaceClient()
        self.confidence_threshold = getattr(settings, 'FACE_RECOGNITION_CONFIDENCE_THRESHOLD', 0.80)
        self.frame_interval = getattr(settings, 'FRAME_SAMPLING_INTERVAL', 0.25)
        self.buffer_before = getattr(settings, 'VIDEO_CLIP_BUFFER_BEFORE', 0.5)
        self.buffer_after = getattr(settings, 'VIDEO_CLIP_BUFFER_AFTER', 0.5)
        self.frame_skip = getattr(settings, 'VIDEO_FRAME_SKIP', 20)  # Analyze every 10th frame for speed
        self.max_workers = getattr(settings, 'VIDEO_PROCESSING_WORKERS', 2)  # Limit concurrent threads
    
    def extract_video_metadata(self, video_path: str) -> Dict:
        """
        Extract basic metadata from video file
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Dict: Video metadata including duration, fps, dimensions
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            print(f"Error extracting video metadata: {str(e)}")
            return {}
    
    def extract_frames_at_intervals(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video at specified intervals and skip frames for speed.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            frame_skip = self.frame_skip
            frame_number = 0
            
            # Resize frame for faster processing
            target_width = 640
            target_height = 480
            print("frame skip :- ",frame_skip)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Extract every Nth frame
                if frame_number % frame_skip == 0:
                    # Resize frame to reduce processing time
                    frame_resized=frame
                    timestamp = frame_number / fps
                    frames.append((timestamp, frame_resized))
                frame_number += 1
            cap.release()
            return frames
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return []
    
    def analyze_video_for_suspects(self, video_path: str, suspect_encodings: List[np.ndarray], progress_callback=None) -> List[Dict]:
        """
        Analyze video for suspect faces with live preview (dev mode).
        """

        detections = []
        frames = self.extract_frames_at_intervals(video_path)
        total_frames = len(frames)
        detections_found = 0
    
        for frame_index, (timestamp, frame) in enumerate(frames):
            if progress_callback:
                progress_callback(frame_index + 1, timestamp, detections_found)
    
            # Detect faces
            detected_faces = self.face_recognition_client.detect_faces_in_frame(frame)
    
            # Draw bounding boxes
            for bounding_box, detection_confidence in detected_faces:
                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            # Show in a window during development
            if getattr(settings, 'DEBUG', False):  # Only show in dev mode
                cv2.imshow("Video Analysis (Dev Preview)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            # Existing detection matching logic
            for bounding_box, detection_confidence in detected_faces:
                x, y, w, h = bounding_box
                
                # Skip very small faces (likely false positives)
                if w < 40 or h < 40:
                    continue
                    
                face_region = frame[y:y+h, x:x+w]
                if face_region.size == 0:
                    continue
    
                temp_dir = tempfile.gettempdir()
                temp_face_path = os.path.join(temp_dir, f"temp_face_{timestamp}_{x}_{y}.jpg")
                cv2.imwrite(temp_face_path, face_region)
    
                try:
                    face_encoding = self.face_recognition_client.extract_features(temp_face_path)
                    for i, suspect_encoding in enumerate(suspect_encodings):
                        similarity = self.face_recognition_client.compare_faces(face_encoding, suspect_encoding)
                        if similarity >= self.confidence_threshold:
                            detection = {
                                'timestamp': timestamp,
                                'confidence': similarity,
                                'bounding_box': bounding_box,
                                'suspect_index': i,
                                'frame_number': int(timestamp * 30)
                            }
                            detections.append(detection)
                            detections_found += 1
                finally:
                    print()
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
    
        cv2.destroyAllWindows()
        return detections

    def analyze_video_for_people(self, video_path: str, progress_callback=None) -> List[Dict]:
        """
        Analyze video for people detection with timestamps
        
        Args:
            video_path (str): Path to video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            List[Dict]: People detection results with timestamps
        """
        detections = []
        frames = self.extract_frames_at_intervals(video_path)
        total_frames = len(frames)
        people_found = 0
        
        for frame_index, (timestamp, frame) in enumerate(frames):
            if progress_callback:
                progress_callback(frame_index + 1, timestamp, people_found)
            
            # Detect people in frame
            detected_people = self.face_recognition_client.detect_people_in_frame(frame)
            
            if detected_people:
                bounding_boxes = []
                confidence_scores = []
                
                # Draw bounding boxes for visualization
                for bounding_box, confidence in detected_people:
                    x, y, w, h = bounding_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {confidence:.2f}', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    bounding_boxes.append(bounding_box)
                    confidence_scores.append(confidence)
                
                # Show in a window during development
                if getattr(settings, 'DEBUG', False):
                    cv2.imshow("People Detection (Dev Preview)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Store detection result
                detection = {
                    'timestamp': timestamp,
                    'frame_number': int(timestamp * 30),  # Estimate frame number
                    'bounding_boxes': bounding_boxes,
                    'confidence_scores': confidence_scores,
                    'person_count': len(bounding_boxes)
                }
                detections.append(detection)
                people_found += len(bounding_boxes)
        
        cv2.destroyAllWindows()
        return detections

    def create_summary_video(self, original_video_path: str, timestamps: List[float], output_path: str) -> bool:
        """
        Create a summary video with clips around detection timestamps
        
        Args:
            original_video_path (str): Path to original video
            timestamps (List[float]): List of detection timestamps
            output_path (str): Path for output summary video
            
        Returns:
            bool: Success status
        """
        print("create_summary_video...")
        try:
            if not timestamps:
                return False

            # Merge overlapping/nearby timestamps
            timestamps_sorted = sorted(timestamps)
            merged_ranges = []
            video = VideoFileClip(original_video_path)
            video_duration = video.duration

            # Buffer values
            buffer_before = self.buffer_before
            buffer_after = self.buffer_after
            print("buffer_before ",buffer_before)
            print("buffer_after ",buffer_after)
            min_gap = buffer_before + buffer_after - 1  # Minimum gap to consider clips separate

            # Build merged ranges
            for ts in timestamps_sorted:
                start = max(0, ts - buffer_before)
                end = min(video_duration, ts + buffer_after)
                if not merged_ranges:
                    merged_ranges.append([start, end])
                else:
                    last_start, last_end = merged_ranges[-1]
                    if start <= last_end:
                        # Overlaps or is close, merge
                        merged_ranges[-1][1] = max(last_end, end)
                    else:
                        merged_ranges.append([start, end])

            clips = []
            for start, end in merged_ranges:
                clip = video.subclip(start, end)
                clips.append(clip)

            if clips:
                final_video = concatenate_videoclips(clips)
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                return True
            video.close()
            return False

        except Exception as e:
            print(f"Error creating summary video: {str(e)}")
            return False
    
    def get_video_thumbnail(self, video_path: str, timestamp: float = 1.0) -> str:
        """
        Extract a thumbnail from video at specified timestamp
        
        Args:
            video_path (str): Path to video file
            timestamp (float): Timestamp for thumbnail extraction
            
        Returns:
            str: Path to thumbnail image
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Generate thumbnail path
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                thumbnail_path = os.path.join(
                    settings.MEDIA_ROOT, 
                    'thumbnails', 
                    f"{video_name}_thumb.jpg"
                )
                
                # Ensure thumbnail directory exists
                os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
                
                # Save thumbnail
                cv2.imwrite(thumbnail_path, frame)
                cap.release()
                
                return thumbnail_path
            
            cap.release()
            return ""
            
        except Exception as e:
            print(f"Error creating thumbnail: {str(e)}")
            return ""
    
    def create_summary_video_from_detections(self, original_video_path: str, detections: List[Dict], output_path: str) -> bool:
        """
        Create a focused summary video showing only detected suspect faces
        with minimal buffer frames around each detection
        
        Args:
            original_video_path (str): Path to original video
            detections (List[Dict]): List of detection dictionaries with frame_number
            output_path (str): Path for output summary video
            
        Returns:
            bool: Success status
        """
        try:
            if not detections:
                return False
            
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                return False
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps
            
            cap.release()
            
            # Convert frame numbers to timestamps with minimal buffer
            # Use very short clips (0.5 seconds before and after detection)
            short_buffer_before = 0.5  # 0.5 seconds before detection
            short_buffer_after = 0.5   # 0.5 seconds after detection
            
            timestamps_with_buffer = []
            for detection in detections:
                frame_number = detection.get('frame_number', 0)
                detection_timestamp = frame_number / fps
                
                # Create short clip around detection
                start_time = max(0, detection_timestamp - short_buffer_before)
                end_time = min(video_duration, detection_timestamp + short_buffer_after)
                
                timestamps_with_buffer.append((start_time, end_time))
            
            # Remove overlapping clips and merge nearby ones
            timestamps_with_buffer.sort(key=lambda x: x[0])
            merged_clips = []
            
            for start, end in timestamps_with_buffer:
                if not merged_clips:
                    merged_clips.append([start, end])
                else:
                    last_start, last_end = merged_clips[-1]
                    # If clips overlap or are very close (within 0.2 seconds), merge them
                    if start <= last_end + 0.2:
                        merged_clips[-1][1] = max(last_end, end)
                    else:
                        merged_clips.append([start, end])
            
            # Create focused summary video using moviepy
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            
            video = VideoFileClip(original_video_path)
            clips = []
            
            for start, end in merged_clips:
                # Keep clips short and focused
                clip_duration = end - start
                if clip_duration > 2.0:  # If merged clip is too long, limit it
                    end = start + 2.0
                
                clip = video.subclip(start, min(end, video.duration))
                clips.append(clip)
            
            if clips:
                final_video = concatenate_videoclips(clips)
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None,
                    preset='fast',  # Faster encoding
                    threads=4
                )
                
                # Clean up
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                
                # Log summary info
                total_duration = sum(end - start for start, end in merged_clips)
                print(f"Created focused summary video: {len(merged_clips)} clips, {total_duration:.1f}s total duration")
                return True
                
            video.close()
            return False
            
        except Exception as e:
            print(f"Error creating focused summary video from detections: {str(e)}")
            return False
    
    def create_individual_detection_clips(self, original_video_path: str, detections: List[Dict], output_path: str) -> bool:
        """
        Create individual short clips for each detection (alternative approach)
        Each clip shows exactly 1 second: 0.3s before + detection frame + 0.7s after
        
        Args:
            original_video_path (str): Path to original video
            detections (List[Dict]): List of detection dictionaries
            output_path (str): Path for output summary video
            
        Returns:
            bool: Success status
        """
        try:
            if not detections:
                return False
                
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                return False
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps
            cap.release()
            
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            
            video = VideoFileClip(original_video_path)
            clips = []
            
            # Create exactly 1-second clips for each detection
            clip_duration = 1.0  # 1 second total
            before_detection = 0.3  # 0.3 seconds before
            after_detection = 0.7   # 0.7 seconds after
            
            processed_timestamps = set()  # Avoid duplicate clips
            
            for detection in detections:
                frame_number = detection.get('frame_number', 0)
                detection_timestamp = frame_number / fps
                
                # Skip if we already have a clip very close to this timestamp
                if any(abs(detection_timestamp - ts) < 0.5 for ts in processed_timestamps):
                    continue
                
                start_time = max(0, detection_timestamp - before_detection)
                end_time = min(video_duration, detection_timestamp + after_detection)
                
                # Ensure minimum clip duration
                if end_time - start_time < 0.8:
                    continue
                
                clip = video.subclip(start_time, end_time)
                clips.append(clip)
                processed_timestamps.add(detection_timestamp)
                
                # Limit number of clips to avoid very long summary videos
                if len(clips) >= 20:  # Maximum 20 clips = 20 seconds summary
                    break
            
            if clips:
                final_video = concatenate_videoclips(clips)
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None,
                    preset='fast',
                    threads=4
                )
                
                # Clean up
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                
                total_duration = len(clips) * clip_duration
                print(f"Created individual detection clips: {len(clips)} clips, {total_duration:.1f}s total")
                return True
                
            video.close()
            return False
            
        except Exception as e:
            print(f"Error creating individual detection clips: {str(e)}")
            return False


def process_video_for_suspects(video_path: str, suspect_encodings: List[np.ndarray]) -> List[Dict]:
    """
    Convenience function to process video for suspect detection
    
    Args:
        video_path (str): Path to video file
        suspect_encodings (List[np.ndarray]): List of suspect face encodings
        
    Returns:
        List[Dict]: Detection results
    """
    processor = VideoProcessor()
    return processor.analyze_video_for_suspects(video_path, suspect_encodings)
