"""
Updated Django views for real-time video streaming using UnifiedVideoProcessor.
This replaces the custom streaming logic with the unified approach from tasks.py
"""

import os
import json
import logging
import asyncio
import tempfile
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import DenyConnection
from .tasks import get_unified_processor
from apps.cases.models import SuspectImage
import cv2
import base64
import numpy as np
import json as json_module
from apps.cases.models import DetectionResult, SuspectImage
from apps.cases.models import ProcessedVideo

from ..cases.models import Case, VideoUpload
from .tasks import (
    get_unified_processor,  # Use singleton to avoid repeated initialization
    get_suspect_data_for_processing,
    process_single_frame_for_websocket
)

logger = logging.getLogger(__name__)

# Global processor instance to avoid repeated initialization during WebSocket sessions
_processor_instance = None

def get_streaming_processor():
    """Get or create a cached processor instance for streaming"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = get_unified_processor()
    return _processor_instance


@login_required
@require_http_methods(["GET"])
def stream_stats(request, case_id: int, video_id: int):
    """Get streaming statistics"""
    try:
        case = get_object_or_404(Case, id=case_id, user=request.user)
        video = get_object_or_404(VideoUpload, id=video_id, case=case)
        
        # For now, return basic stats
        # In a real implementation, you might track these in Redis or similar
        stats = {
            'processed_frames': 0,
            'faces_detected': 0,
            'suspects_detected': 0,
            'fps': 30,
            'is_streaming': False
        }
        
        return JsonResponse({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def stop_stream(request, case_id: int, video_id: int):
    """Stop streaming for a specific video"""
    try:
        case = get_object_or_404(Case, id=case_id, user=request.user)
        video = get_object_or_404(VideoUpload, id=video_id, case=case)
        
        # Mark video as processed if not already
        if not video.processed:
            video.processed = True
            video.processing_completed_at = timezone.now()
            video.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Stream stopped'
        })
        
    except Exception as e:
        logger.error(f"Stop stream error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


class VideoStreamConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer using UnifiedVideoProcessor for consistent processing
    """
    
    async def connect(self):
        """Handle WebSocket connection"""
        try:
            # Get parameters from URL
            self.case_id = self.scope['url_route']['kwargs']['case_id']
            self.video_id = self.scope['url_route']['kwargs']['video_id']
            self.user = self.scope['user']
            
            # Check authentication
            if not self.user.is_authenticated:
                await self.close()
                return
            
            # Validate access to case and video
            try:
                case = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: Case.objects.get(id=self.case_id, user=self.user)
                )
                video = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: VideoUpload.objects.get(id=self.video_id, case=case)
                )
                
                self.case = case
                self.video = video
                
            except Exception as e:
                logger.error(f"Invalid case/video access: {e}")
                await self.close()
                return
            
            # Accept connection
            await self.accept()
            
            # Setup and start processing
            await self.setup_processing()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Clean up temporary files
        if hasattr(self, 'temp_video_path') and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")
        
        logger.info(f"WebSocket disconnected: {close_code}")
    
    async def receive(self, text_data):
        """Handle messages from WebSocket"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'start_stream':
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'message': 'Streaming started'
                }))
                
            elif message_type == 'stop_stream':
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'message': 'Streaming stopped'
                }))
                await self.close()
                
            elif message_type == 'get_stats':
                # Send current statistics
                stats = {
                    'processed_frames': getattr(self, 'frame_count', 0),
                    'faces_detected': getattr(self, 'total_faces', 0),
                    'suspects_detected': getattr(self, 'total_suspects', 0),
                    'fps': 30
                }
                await self.send(text_data=json.dumps({
                    'type': 'stats',
                    'data': stats
                }))
                
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
    
    async def setup_processing(self):
        """Setup video processing with UnifiedVideoProcessor"""
        try:
            # Get suspect data
            suspect_encodings, suspect_mapping = await asyncio.get_event_loop().run_in_executor(
                None, get_suspect_data_for_processing, self.case_id
            )
            
            self.suspect_encodings = suspect_encodings
            self.suspect_mapping = suspect_mapping
            
            # Create temporary video file
            temp_video_path = await asyncio.get_event_loop().run_in_executor(
                None, self.video.write_temp_file
            )
            self.temp_video_path = temp_video_path
            
            # Start processing
            await self.start_processing()
            
        except Exception as e:
            logger.error(f"Setup processing error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Setup error: {str(e)}'
            }))
            await self.close()
    
    async def start_processing(self):
        """Start ultra-fast video processing optimized for WebSocket streaming"""
        try:
            # Get suspects
            suspects = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(self.case.suspect_images.filter(processed=True))
            )
            if not suspects:
                await self.send_error("No processed suspects found in case")
                return
            
            # Get processor singleton
            processor = get_unified_processor()
            
            # Extract suspect encodings
            suspect_encodings = []
            suspect_mapping = {}
            
            for i, suspect in enumerate(suspects):
                if suspect.face_encoding:
                    encoding = json_module.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                    suspect_encodings.append(np.array(encoding))
                    suspect_mapping[i] = suspect
            
            if not suspect_encodings:
                await self.send_error("No valid suspect encodings found")
                return
            
            # Process video with optimized parameters
            temp_dir = tempfile.gettempdir()
            video_path = os.path.join(temp_dir, f"temp_video_{self.video.id}.{self.video.video_type}")
            
            # Save video to temp file
            video_data = await asyncio.get_event_loop().run_in_executor(
                None, self.video.get_video_data
            )
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {video_path}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Ultra-fast processing: Skip many frames (every 60 frames = ~2 seconds)
                frame_skip = 10
                frame_number = 0
                detections = []
                frames_sent = 0
                
                await self.send(text_data=json_module.dumps({
                    'type': 'status',
                    'message': f'Starting fast processing: {total_frames} frames, processing every {frame_skip} frames'
                }))
                
                while cap.isOpened():  # Limit to 20 frames max
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for speed
                    if frame_number % frame_skip != 0:
                        frame_number += 1
                        continue
                    
                    timestamp = frame_number / fps if fps > 0 else 0
                    
                    # Resize frame for speed (320p processing)
                    height, width = frame.shape[:2]
                    if width > 320:
                        scale = 320 / width
                        new_width = 320
                        new_height = int(height * scale)
                        frame_small = cv2.resize(frame, (new_width, new_height))
                    else:
                        frame_small = frame
                        scale = 1.0
                    
                    # Detect faces
                    detected_faces = processor.face_client.detect_faces_in_frame(frame_small)
                    
                    if detected_faces:
                        frames_sent += 1
                        frame_display = frame.copy()
                        suspects_found = 0
                        
                        # Process first face only for speed
                        for bounding_box, confidence in detected_faces:  # Only first face
                            if scale != 1.0:
                                x, y, w, h = [int(coord / scale) for coord in bounding_box]
                            else:
                                x, y, w, h = bounding_box
                            
                            if w < 60 or h < 60:
                                continue
                            
                            # Draw detection
                            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Quick face recognition
                            try:
                                face_region = frame[y:y+h, x:x+w]
                                if face_region.size > 0:
                                    face_encoding = processor.face_client.extract_features_from_array(face_region)
                                    
                                    # Compare with suspects
                                    for i, suspect_encoding in enumerate(suspect_encodings):
                                        similarity = processor.face_client.compare_faces(face_encoding, suspect_encoding)
                                        
                                        if similarity >= 0.8:  # Lower threshold for speed
                                            suspects_found += 1
                                            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 3)
                                            cv2.putText(frame_display, 'SUSPECT', (x, y-10), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                            
                                            detections.append({
                                                'timestamp': timestamp,
                                                'confidence': similarity,
                                                'suspect_index': i,
                                                'frame_number': frame_number
                                            })
                                            
                                            # Store detection in database for results page
                                            def save_detection():
                                                try:
                                                    # Get the suspect image
                                                    suspect_images = list(self.case.suspect_images.all())
                                                    if i < len(suspect_images):
                                                        suspect = suspect_images[i]
                                                        
                                                        # Create detection result
                                                        DetectionResult.objects.create(
                                                            video=self.video,
                                                            suspect=suspect,
                                                            timestamp=timestamp,
                                                            confidence=similarity,
                                                            frame_number=frame_number,
                                                            bounding_box=[int(x), int(y), int(w), int(h)]
                                                        )
                                                except Exception as e:
                                                    logger.error(f"Error saving detection: {e}")
                                            
                                            # Run database save in executor (don't await it to keep processing fast)
                                            asyncio.get_event_loop().run_in_executor(None, save_detection)
                                            
                                            break
                            except Exception as e:
                                print(f"Face recognition error: {str(e)}")
                        
                        # Send frame (compressed)
                        height, width = frame_display.shape[:2]
                        if width > 480:
                            display_frame = cv2.resize(frame_display, (480, int(height * 480 / width)))
                        else:
                            display_frame = frame_display
                        
                        _, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        progress = (frame_number / total_frames * 100) if total_frames > 0 else 0
                        
                        await self.send(text_data=json_module.dumps({
                            'type': 'frame',
                            'data': frame_b64,
                            'timestamp': timestamp,
                            'frame_number': frame_number,
                            'faces_detected': len(detected_faces),
                            'suspects_detected': suspects_found,
                            'stats': {
                                'progress': progress,
                                'processed_frames': frames_sent,
                                'total_frames': total_frames,
                                'total_detections': len(detections)
                            }
                        }))
                        
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.1)
                    
                    frame_number += 1
                
                cap.release()
                
                # Generate summary video if detections found
                if detections:
                    await self.send(text_data=json_module.dumps({
                        'type': 'status',
                        'message': f'Generating summary video from {len(detections)} detections...'
                    }))
                    
                    try:
                        # Create summary video
                        summary_video_path = os.path.join(temp_dir, f"summary_{self.video.id}.mp4")
                        
                        # Use processor to create focused summary video
                        success = await asyncio.get_event_loop().run_in_executor(
                            None,
                            processor.create_summary_video_from_detections,
                            video_path,
                            detections,
                            summary_video_path
                        )
                        
                        # If focused summary fails, try individual clips approach
                        if not success:
                            await self.send(text_data=json_module.dumps({
                                'type': 'status',
                                'message': 'Trying alternative summary creation method...'
                            }))
                            
                            success = await asyncio.get_event_loop().run_in_executor(
                                None,
                                processor.create_individual_detection_clips,
                                video_path,
                                detections,
                                summary_video_path
                            )
                        
                        if success and os.path.exists(summary_video_path):
                            # Read summary video data
                            with open(summary_video_path, 'rb') as f:
                                summary_data = f.read()
                            
                            # Save to database
                            
                            def save_summary_video():
                                processed_video, created = ProcessedVideo.objects.get_or_create(
                                    original_video=self.video,
                                    case=self.case,
                                    defaults={
                                        'processed_data': summary_data,
                                        'processed_name': f"summary_{self.video.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                        'processed_type': 'mp4',
                                        'processed_size': len(summary_data),
                                        'total_detections': len(detections),
                                        'summary_duration': 0.0,  # Will be calculated by video processor if needed
                                    }
                                )
                                
                                if not created:
                                    processed_video.processed_data = summary_data
                                    processed_video.processed_size = len(summary_data)
                                    processed_video.total_detections = len(detections)
                                    processed_video.save()
                                
                                return processed_video
                            
                            processed_video = await asyncio.get_event_loop().run_in_executor(
                                None, save_summary_video
                            )
                            
                            # Clean up temp summary video
                            os.remove(summary_video_path)
                            
                            # Send summary completion message
                            await self.send(text_data=json_module.dumps({
                                'type': 'summary_complete',
                                'total_detections': len(detections),
                                'suspects_found': len(set(d['suspect_index'] for d in detections)),
                                'summary_video_id': processed_video.id,
                                'message': 'Summary video generated successfully!'
                            }))
                            
                        else:
                            await self.send(text_data=json_module.dumps({
                                'type': 'status',
                                'message': 'Warning: Could not generate summary video'
                            }))
                            
                    except Exception as summary_error:
                        logger.error(f"Summary video generation error: {summary_error}")
                        await self.send(text_data=json_module.dumps({
                            'type': 'status',
                            'message': f'Summary video generation failed: {str(summary_error)}'
                        }))
                
                # Mark video as processed
                def mark_processed():
                    self.video.processed = True
                    self.video.processing_completed_at = timezone.now()
                    self.video.save()
                    
                    self.case.status = 'completed'
                    self.case.save()
                
                await asyncio.get_event_loop().run_in_executor(None, mark_processed)
                
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
            
            # Send completion
            await self.send(text_data=json_module.dumps({
                'type': 'completed',
                'message': f'Fast processing completed! Found {len(detections)} detections.',
                'total_detections': len(detections),
                'frames_sent': frames_sent
            }))
                
        except Exception as e:
            logger.error(f"WebSocket processing error: {e}")
            await self.send_error(f"Processing failed: {str(e)}")

    async def send_error(self, error_message: str):
        """Send error message to WebSocket client"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': error_message
        }))
    
    async def send_frame_data(self, frame_data):
        """Send frame data to WebSocket client"""
        try:
            await self.send(text_data=json.dumps(frame_data))
        except Exception as e:
            logger.error(f"Error sending frame data: {e}")
    
    def mark_video_processed(self):
        """Mark video as processed in database"""
        try:
            self.video.processed = True
            self.video.processing_completed_at = timezone.now()
            self.video.save()
            
            case = self.video.case
            case.status = 'completed'
            case.save()
            
        except Exception as e:
            logger.error(f"Error marking video processed: {e}")

