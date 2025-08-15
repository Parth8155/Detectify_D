"""
Django views for real-time video streaming with face detection.
Supports both MJPEG and WebSocket streaming.
"""

import os
import json
import logging
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse, Http404
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils import timezone
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import DenyConnection
import asyncio
import threading
from typing import AsyncGenerator

from ..cases.models import Case, VideoUpload
from .realtime_processor import streaming_sessions, VideoStreamProcessor

logger = logging.getLogger(__name__)


def _get_suspect_data(case_id: int):
    """Helper function to get suspect encodings and mapping for a case.
    
    Args:
        case_id: Case ID
        
    Returns:
        Tuple of (suspect_encodings, suspect_mapping)
    """
    try:
        from ..cases.models import SuspectImage
        import json
        import numpy as np
        
        suspects = SuspectImage.objects.filter(case_id=case_id, processed=True)
        suspect_encodings = []
        suspect_mapping = {}
        
        for i, suspect in enumerate(suspects):
            if suspect.face_encoding:
                try:
                    encoding = json.loads(suspect.face_encoding) if isinstance(suspect.face_encoding, str) else suspect.face_encoding
                    suspect_encodings.append(np.array(encoding))
                    suspect_mapping[i] = suspect
                except Exception as e:
                    logger.error(f"Error loading suspect encoding for suspect {suspect.id}: {e}")
        
        logger.info(f"Loaded {len(suspect_encodings)} suspect encodings for case {case_id}")
        return suspect_encodings, suspect_mapping
    
    except Exception as e:
        logger.error(f"Error loading suspect data for case {case_id}: {e}")
        return [], {}


def _mark_video_processed_sync(case_id: int, video_id: int):
    """Synchronously mark a VideoUpload as processed and update the Case status.

    This helper is safe to call from background threads or executors.
    """
    try:
        from ..cases.models import Case, VideoUpload

        video = VideoUpload.objects.filter(id=video_id, case_id=case_id).first()
        if not video:
            logger.debug(f"_mark_video_processed_sync: video not found case={case_id} video={video_id}")
            return

        if not video.processed:
            video.processed = True
            video.processing_completed_at = timezone.now()
            video.save()

            try:
                case = video.case
                case.status = 'completed'
                case.save()
            except Exception:
                logger.exception('Failed to update case status after marking video processed')
        else:
            logger.debug(f"_mark_video_processed_sync: video already marked processed id={video_id}")
    except Exception:
        logger.exception('Error marking video as processed')

@login_required
@require_http_methods(["GET"])
def stream_stats(request, case_id: int, video_id: int):
    """Get streaming statistics for a video.
    
    Args:
        request: Django request object
        case_id: Case ID
        video_id: Video ID
        
    Returns:
        JsonResponse with streaming statistics
    """
    try:
        # Get video object
        case = get_object_or_404(Case, id=case_id, user=request.user)
        video = get_object_or_404(VideoUpload, id=video_id, case=case)
        
        # Get session
        session_id = f"mjpeg_{case_id}_{video_id}_{request.user.id}"
        processor = streaming_sessions.get_session(session_id)
        
        if not processor:
            session_id = f"ws_{case_id}_{video_id}_{request.user.id}"
            processor = streaming_sessions.get_session(session_id)
        
        if processor:
            stats = processor.get_stats()
            return JsonResponse({
                'status': 'success',
                'stats': stats,
                'is_streaming': processor.is_streaming
            })
        else:
            return JsonResponse({
                'status': 'success',
                'stats': {
                    'total_frames': 0,
                    'processed_frames': 0,
                    'faces_detected': 0,
                    'fps': 0,
                    'progress': 0,
                    'avg_faces_per_frame': 0
                },
                'is_streaming': False
            })
            
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
@csrf_exempt
def stop_stream(request, case_id: int, video_id: int):
    """Stop streaming for a video.
    
    Args:
        request: Django request object
        case_id: Case ID
        video_id: Video ID
        
    Returns:
        JsonResponse with operation status
    """
    try:
        # Get video object
        case = get_object_or_404(Case, id=case_id, user=request.user)
        video = get_object_or_404(VideoUpload, id=video_id, case=case)
        
        # Stop sessions
        session_ids = [
            f"mjpeg_{case_id}_{video_id}_{request.user.id}",
            f"ws_{case_id}_{video_id}_{request.user.id}"
        ]
        
        for session_id in session_ids:
            processor = streaming_sessions.get_session(session_id)
            if processor:
                processor.stop_streaming()
                streaming_sessions.remove_session(session_id)
        
        return JsonResponse({
            'status': 'success',
            'message': 'Streaming stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Stop stream error: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


class VideoStreamConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time video streaming."""
    
    async def connect(self):
        """Handle WebSocket connection."""
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
                from ..cases.models import Case, VideoUpload
                case = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: Case.objects.get(id=self.case_id, user=self.user)
                )
                video = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: VideoUpload.objects.get(id=self.video_id, case=case)
                )
            except Exception as e:
                logger.error(f"Invalid case/video access: {e}")
                await self.close()
                return
            
            # Write temp video file for streaming (run in executor)
            loop = asyncio.get_event_loop()
            try:
                temp_video_path = await loop.run_in_executor(None, video.write_temp_file)
                # Get suspect data for recognition
                suspect_encodings, suspect_mapping = await loop.run_in_executor(
                    None, _get_suspect_data, self.case_id
                )
                # Store for use in streaming
                self.suspect_encodings = suspect_encodings
                self.suspect_mapping = suspect_mapping
                self.video = video
            except Exception as e:
                logger.error(f"Failed to write temp video file for WebSocket: {e}")
                await self.close()
                return

            # Accept connection
            await self.accept()

            # Create session
            self.session_id = f"ws_{self.case_id}_{self.video_id}_{self.user.id}"
            self.temp_video_path = temp_video_path

            # Start streaming in background task
            self.streaming_task = asyncio.create_task(self.start_streaming(self.temp_video_path))
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        try:
            # Cancel streaming task
            if hasattr(self, 'streaming_task'):
                self.streaming_task.cancel()

            # Remove session
            if hasattr(self, 'session_id'):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    streaming_sessions.remove_session,
                    self.session_id
                )

            # Cleanup temp video file if created
            if hasattr(self, 'temp_video_path'):
                try:
                    await asyncio.get_event_loop().run_in_executor(None, lambda: os.remove(self.temp_video_path) if os.path.exists(self.temp_video_path) else None)
                except Exception:
                    logger.exception('Failed to remove temp video file during disconnect')
                
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")
    
    async def receive(self, text_data):
        """Handle messages from WebSocket."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'start_stream':
                # Stream control - already handled in connect
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'message': 'Streaming started'
                }))
                
            elif message_type == 'stop_stream':
                # Stop streaming
                if hasattr(self, 'streaming_task'):
                    self.streaming_task.cancel()
                
                await self.send(text_data=json.dumps({
                    'type': 'status',
                    'message': 'Streaming stopped'
                }))
                
            elif message_type == 'get_stats':
                # Get current statistics
                processor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    streaming_sessions.get_session,
                    self.session_id
                )
                
                if processor:
                    stats = await asyncio.get_event_loop().run_in_executor(
                        None,
                        processor.get_stats
                    )
                    await self.send(text_data=json.dumps({
                        'type': 'stats',
                        'data': stats
                    }))
                    
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
    
    async def start_streaming(self, video_path: str):
        """Start video streaming with face detection.
        
        Args:
            video_path: Path to the video file
        """
        try:
            # Create processor
            detection_model_path = os.path.join(settings.BASE_DIR, "face_detection_yunet_2023mar.onnx")
            
            processor = await asyncio.get_event_loop().run_in_executor(
                None,
                streaming_sessions.create_session,
                self.session_id,
                video_path,
                detection_model_path,
                self.video,  # Pass video object for saving DetectionResult
                self.suspect_encodings,
                self.suspect_mapping
            )
            
            # Stream frames
            def frame_generator():
                return processor.generate_websocket_frames(target_fps=30, quality=80)
            
            frame_iter = await asyncio.get_event_loop().run_in_executor(
                None,
                frame_generator
            )
            
            # Send frames to WebSocket
            for frame_data in frame_iter:
                if self.streaming_task.cancelled():
                    break
                    
                await self.send(text_data=json.dumps(frame_data))
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
            
            # Send completion message
            await self.send(text_data=json.dumps({
                'type': 'completed',
                'message': 'Video processing completed'
            }))

            # Save any pending detections to database before marking complete
            def _save_detections():
                try:
                    processor = streaming_sessions.get_session(self.session_id)
                    if processor and hasattr(processor, 'save_pending_detections'):
                        saved_count = processor.save_pending_detections()
                        logger.info(f"Saved {saved_count} pending detections for video {self.video_id}")
                        return saved_count
                    return 0
                except Exception as e:
                    logger.error(f"Error saving pending detections: {e}")
                    return 0

            # Save detections in executor to avoid async issues
            try:
                saved_count = await asyncio.get_event_loop().run_in_executor(None, _save_detections)
                logger.info(f"Detection save completed: {saved_count} detections saved")
            except Exception:
                logger.exception('Failed to save pending detections after WebSocket stream')

            # Mark video processed in DB (run in executor so it's synchronous DB call)
            try:
                await asyncio.get_event_loop().run_in_executor(None, _mark_video_processed_sync, int(self.case_id), int(self.video_id))
            except Exception:
                logger.exception('Failed to mark video processed after WebSocket stream')

            # Note: We don't need to run additional batch processing here since 
            # DetectionResult objects are already being saved during streaming
            # This makes the results immediately available on the results page
            logger.info(f"WebSocket streaming completed for video {self.video_id}. Results should be immediately available.")
            
        except asyncio.CancelledError:
            logger.info("Streaming task cancelled")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Streaming error: {str(e)}'
            }))
