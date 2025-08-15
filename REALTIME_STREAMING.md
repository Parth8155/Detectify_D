# Real-time Video Processing with Face Detection

This document describes the real-time video streaming system that processes videos in parallel with live UI updates and streams processed frames to a browser.

## System Features

### 1. Real-time Face Detection
- Uses OpenCV's YuNet face detection model
- Draws green rectangles around detected faces
- Provides confidence scores for each detection

### 2. Two Streaming Methods
1. **MJPEG Streaming**: HTTP multipart/x-mixed-replace format
2. **WebSocket Streaming**: Base64 encoded JPEGs over WebSockets

### 3. Live UI Updates
- Real-time progress bars
- Frame-by-frame processing status
- Face detection counts
- Processing statistics

## How to Use

### 1. Start the System
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

### 2. Access the Interface
1. Navigate to `http://127.0.0.1:8000/dashboard/`
2. Select a case with uploaded videos
3. Click "Real-time Stream" next to any video

### 3. Streaming Options
- **MJPEG Stream**: Click "Start MJPEG Stream" for HTTP-based streaming
- **WebSocket Stream**: Click "Start WebSocket Stream" for real-time updates
- **Settings**: Adjust quality (60%-95%) and FPS (15-60)

### 4. Background Processing
When you click "Process Video" on the main case page:
- Video processing starts immediately in the background
- UI shows real-time progress updates every 2 seconds
- Processing continues even if you navigate away
- Progress is maintained across page reloads

## Technical Implementation

### Backend Components

#### 1. Real-time Processor (`apps/media_processing/realtime_processor.py`)
```python
class VideoStreamProcessor:
    - Handles video file reading
    - Performs face detection on each frame
    - Manages frame encoding and streaming
    - Tracks processing statistics
```

#### 2. Streaming Views (`apps/media_processing/streaming_views.py`)
```python
# MJPEG Streaming
def stream_video_mjpeg(request, case_id, video_id):
    # Returns HTTP streaming response

# WebSocket Consumer
class VideoStreamConsumer(AsyncWebsocketConsumer):
    # Handles WebSocket connections and frame streaming
```

#### 3. Session Management
```python
class StreamingSession:
    - Manages multiple concurrent streaming sessions
    - Handles session cleanup
    - Prevents resource leaks
```

### Frontend Components

#### 1. Real-time Interface (`templates/cases/stream_video.html`)
- Dual display support (MJPEG img tag + WebSocket canvas)
- Real-time statistics panel
- Stream controls and settings
- Responsive design with cybersecurity theme

#### 2. Progress Monitoring (`templates/cases/case_detail.html`)
- Enhanced JavaScript for real-time progress tracking
- Polling-based status updates
- Dynamic UI element creation
- Processing state management

## API Endpoints

### Streaming Endpoints
```
/dashboard/{case_id}/video/{video_id}/stream/          # MJPEG stream
/dashboard/{case_id}/video/{video_id}/stream/page/     # Streaming interface
/dashboard/{case_id}/video/{video_id}/stream/stats/    # Statistics API
/dashboard/{case_id}/video/{video_id}/stream/stop/     # Stop streaming
```

### WebSocket Endpoint
```
ws://127.0.0.1:8000/ws/stream/{case_id}/{video_id}/    # WebSocket streaming
```

## Configuration

### Django Settings
```python
# Add to INSTALLED_APPS
'channels',

# ASGI Configuration
ASGI_APPLICATION = 'detectify_project.routing.application'
```

### Face Detection Model
- Download YuNet model: `face_detection_yunet_2023mar.onnx`
- Place in project root directory
- Model provides high-accuracy face detection

## Performance Considerations

### 1. Frame Rate Control
- Default: 30 FPS
- Configurable: 15-60 FPS
- Higher FPS = more CPU usage

### 2. Quality Settings
- Low (60%): Faster processing, lower quality
- Medium (80%): Balanced performance
- High (95%): Best quality, slower processing

### 3. Memory Management
- Frame buffer: 30 frames maximum
- Automatic session cleanup
- Memory-efficient JPEG encoding

### 4. Concurrent Processing
- Multiple users can stream simultaneously
- Session isolation prevents interference
- Resource limits prevent server overload

## Monitoring and Debugging

### 1. Server Logs
```
Processing frame 123/1662 (7.4%) - 9 detections at 20.50s
[GET /dashboard/15/video/18/status/] - Real-time status updates
```

### 2. Browser Console
- WebSocket connection status
- Frame reception logs
- Error messages and debugging info

### 3. Statistics Tracking
- Total frames processed
- Processing speed (FPS)
- Face detection count
- Average faces per frame
- Processing time per frame

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Solution: Download YuNet model to project root
   URL: https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   ```

2. **WebSocket Connection Failed**
   ```
   Solution: Ensure Channels is installed and configured
   pip install channels channels-redis
   ```

3. **Streaming Stops Unexpectedly**
   ```
   Solution: Check browser console for errors
   Verify video file exists and is accessible
   ```

4. **Poor Performance**
   ```
   Solution: Reduce FPS or quality settings
   Check CPU usage and available memory
   ```

### Testing the System
Run the test script to verify all components:
```bash
python test_streaming.py
```

## Security Considerations

1. **Authentication**: All endpoints require user login
2. **Authorization**: Users can only access their own cases
3. **Resource Limits**: Automatic session cleanup prevents resource exhaustion
4. **Input Validation**: File type and size restrictions

## Future Enhancements

1. **Multi-face Recognition**: Match detected faces against suspect database
2. **Video Analytics**: Motion detection, object tracking
3. **Export Features**: Save processed video with annotations
4. **Real-time Alerts**: Notifications when specific faces are detected
5. **Mobile Support**: Responsive design for mobile devices

## Development Notes

- Uses Django Channels for WebSocket support
- OpenCV for video processing and face detection
- Pillow for image manipulation
- NumPy for efficient array operations
- JavaScript for real-time UI updates

This system provides a comprehensive solution for real-time video processing with face detection, suitable for security and surveillance applications.
