from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


class Case(models.Model):
    """Model for managing investigation cases"""
    
    STATUS_CHOICES = [
        ('created', 'Created'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cases')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='created')
    list_display = ('id', 'name', 'user', 'status', 'created_at')
    readonly_fields = ('id',)

    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.user.username}"


class SuspectImage(models.Model):
    """Model for storing suspect images and their face encodings"""
    
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='suspect_images')
    # Store image data directly in database
    image_data = models.BinaryField()
    image_name = models.CharField(max_length=255)
    image_type = models.CharField(max_length=10, choices=[('jpg', 'JPEG'), ('png', 'PNG'), ('jpeg', 'JPEG')])
    image_size = models.IntegerField()  # File size in bytes
    face_encoding = models.JSONField(null=True, blank=True)  # Store 4096-dim array
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Suspect image for {self.case.name}"
    
    def save_image_from_file(self, image_file):
        """Save image file data to database"""
        self.image_data = image_file.read()
        self.image_name = image_file.name
        self.image_size = len(self.image_data)
        # Determine image type from filename
        if self.image_name.lower().endswith('.png'):
            self.image_type = 'png'
        elif self.image_name.lower().endswith(('.jpg', '.jpeg')):
            self.image_type = 'jpg'
        self.save()
    
    def get_image_data(self):
        """Get image data as bytes"""
        return self.image_data


class VideoUpload(models.Model):
    """Model for uploaded videos to be processed"""
    
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='videos')
    # Store video data directly in database
    video_data = models.BinaryField()
    video_name = models.CharField(max_length=255)
    video_type = models.CharField(max_length=10, choices=[('mp4', 'MP4'), ('avi', 'AVI'), ('mkv', 'MKV'), ('mov', 'MOV')])
    video_size = models.IntegerField()  # File size in bytes
    duration = models.FloatField(null=True, blank=True)  # in seconds
    fps = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Video for {self.case.name}"
    
    def save_video_from_file(self, video_file):
        """Save video file data to database"""
        self.video_data = video_file.read()
        self.video_name = video_file.name
        self.video_size = len(self.video_data)
        # Determine video type from filename
        extension = self.video_name.lower().split('.')[-1]
        if extension in ['mp4', 'avi', 'mkv', 'mov']:
            self.video_type = extension
        else:
            self.video_type = 'mp4'  # Default
        self.save()
    
    def get_video_data(self):
        """Get video data as bytes"""
        return self.video_data
    
    def write_temp_file(self):
        """Write video data to a temporary file and return the path"""
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_video_{self.id}.{self.video_type}")
        with open(temp_path, 'wb') as f:
            f.write(self.video_data)
        return temp_path


class DetectionResult(models.Model):
    """Model for storing face detection results"""
    
    video = models.ForeignKey(VideoUpload, on_delete=models.CASCADE, related_name='detections')
    suspect = models.ForeignKey(SuspectImage, on_delete=models.CASCADE, related_name='detections')
    timestamp = models.FloatField()  # in seconds
    confidence = models.FloatField()
    frame_number = models.IntegerField()
    bounding_box = models.JSONField()  # [x, y, width, height]
    created_at = models.DateTimeField(auto_now_add=True)
    list_display = ('id', 'video', 'suspect', 'timestamp', 'confidence')
    readonly_fields = ('id',)

    class Meta:
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['video', 'timestamp']),
            models.Index(fields=['confidence']),
        ]
    
    def __str__(self):
        return f"Detection at {self.timestamp}s (confidence: {self.confidence:.2f})"


class ProcessedVideo(models.Model):
    """Model for storing processed/summary videos"""
    
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='processed_videos')
    original_video = models.ForeignKey(VideoUpload, on_delete=models.CASCADE)
    # Store processed video data directly in database
    processed_data = models.BinaryField()
    processed_name = models.CharField(max_length=255)
    processed_type = models.CharField(max_length=10, default='mp4')
    processed_size = models.IntegerField()  # File size in bytes
    total_detections = models.IntegerField(default=0)
    summary_duration = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Processed video for {self.case.name}"
    
    def save_processed_from_file(self, video_path, filename):
        """Save processed video file data to database"""
        with open(video_path, 'rb') as f:
            self.processed_data = f.read()
        self.processed_name = filename
        self.processed_size = len(self.processed_data)
        self.processed_type = 'mp4'
        self.save()
    
    def get_processed_data(self):
        """Get processed video data as bytes"""
        return self.processed_data


class PeopleDetection(models.Model):
    """Model for storing people detection results in videos"""
    
    video = models.ForeignKey(VideoUpload, on_delete=models.CASCADE, related_name='people_detections')
    timestamp = models.FloatField()  # in seconds
    frame_number = models.IntegerField()
    person_count = models.IntegerField()  # Number of people detected in this frame
    bounding_boxes = models.JSONField()  # List of bounding boxes for each person
    confidence_scores = models.JSONField()  # List of confidence scores for each detection
    person_ids = models.JSONField()  # Track person across frames (simple ID assignment)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['video', 'timestamp']),
            models.Index(fields=['person_count']),
        ]
    
    def __str__(self):
        return f"People detection at {self.timestamp}s ({self.person_count} people)"
