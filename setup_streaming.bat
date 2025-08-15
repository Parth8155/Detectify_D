@echo off
echo Installing Real-time Video Streaming Dependencies...
echo ===================================================

echo.
echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Checking for YuNet face detection model...
if not exist "face_detection_yunet_2023mar.onnx" (
    echo YuNet model not found. Please download it manually from:
    echo https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    echo.
    echo Or use this direct link:
    echo https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    echo.
) else (
    echo ✓ YuNet model found
)

echo.
echo Running Django migrations...
python manage.py migrate

echo.
echo Testing the streaming system...
python test_streaming.py

echo.
echo Setup complete! You can now:
echo 1. Start the development server: python manage.py runserver
echo 2. Navigate to a case with videos
echo 3. Click "Real-time Stream" on any video
echo.
pause
