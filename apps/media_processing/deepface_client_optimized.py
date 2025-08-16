"""
Face Recognition Client using DeepFace with Singleton Pattern
Implements the FacialRecognition interface with optimized model initialization
"""

import numpy as np
import cv2
import os
import time
import tempfile
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class FacialRecognition(ABC):
    """Abstract base class for facial recognition implementations"""
    
    @abstractmethod
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract facial features from an image"""
        pass
    
    @abstractmethod
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Compare two face encodings and return similarity score"""
        pass


class DeepFaceModelSingleton:
    """Singleton class to ensure only one instance of DeepFace models"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepFaceModelSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model_name = "DeepFace"
            self.input_shape = (152, 152)
            self.output_shape = 4096
            self._initialize_deepface()
            self._initialize_yunet()
            self._initialize_people_detector()
            DeepFaceModelSingleton._initialized = True
    
    def _initialize_deepface(self):
        """Initialize the DeepFace model - only once"""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print(f"DeepFace model '{self.model_name}' initialized successfully")
        except ImportError as e:
            print("DeepFace import failed:", e)
            raise
        except Exception as e:
            print("An unexpected error occurred while importing DeepFace:", e)
            raise
    
    def _initialize_yunet(self):
        """Initialize YuNet face detector - only once"""
        try:
            from django.conf import settings
            model_path = os.path.join(settings.BASE_DIR, "face_detection_yunet_2023mar.onnx")
            
            if not os.path.exists(model_path):
                print(f"YuNet model not found at {model_path}, trying temp directory...")
                # Try temp directory
                model_path = os.path.join(tempfile.gettempdir(), 'face_detection_yunet_2023mar.onnx')
                
                if not os.path.exists(model_path):
                    print("Downloading YuNet model...")
                    import urllib.request
                    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                    urllib.request.urlretrieve(url, model_path)
                    print("YuNet model downloaded successfully")
                
            self.yunet_detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (0, 0),  # Will be set dynamically
                0.6,  # Score threshold
                0.3,  # NMS threshold
                5000  # Top K
            )
            print("YuNet face detector initialized successfully")
            
            # Fallback cascade
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        except Exception as e:
            print(f"Error initializing YuNet: {e}")
            self.yunet_detector = None
            # Fallback to opencv cascade
            try:
                self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Fallback: Using Haar cascade detector")
            except Exception as cascade_e:
                print(f"Error loading cascade: {str(cascade_e)}")
                self.cascade = None
    
    def _initialize_people_detector(self):
        """Initialize HOG people detector - only once"""
        try:
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("HOG people detector initialized successfully")
        except Exception as e:
            print(f"Error initializing HOG detector: {e}")
            self.hog_detector = None


class DeepFaceClient(FacialRecognition):
    """
    DeepFace implementation for face recognition using singleton pattern
    Uses DeepFace model with 4096-dimensional feature extraction
    """
    
    def __init__(self):
        # Get the singleton instance
        self._singleton = DeepFaceModelSingleton()
        
        # Expose singleton properties
        self.model_name = self._singleton.model_name
        self.input_shape = self._singleton.input_shape
        self.output_shape = self._singleton.output_shape
        self.deepface = self._singleton.deepface
        self.yunet_detector = self._singleton.yunet_detector
        self.cascade = self._singleton.cascade
        self.hog_detector = self._singleton.hog_detector
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract facial features from an image using DeepFace
        Returns 4096-dimensional feature vector
        """
        try:
            # Use DeepFace to extract features
            embedding = self.deepface.represent(
                img_path=image_path,
                model_name='VGG-Face',
                enforce_detection=False
            )
            
            # Extract the embedding array
            features = np.array(embedding[0]['embedding'])
            
            # Ensure consistent output shape
            if len(features) != self.output_shape:
                if len(features) < self.output_shape:
                    # Pad with zeros if too small
                    features = np.pad(features, (0, self.output_shape - len(features)))
                else:
                    # Truncate if too large
                    features = features[:self.output_shape]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.output_shape)
    
    def extract_features_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract facial features directly from numpy array (faster for WebSocket)
        Returns 4096-dimensional feature vector
        """
        try:
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Use DeepFace to extract features from array
            embedding = self.deepface.represent(
                img_path=image_rgb,  # DeepFace can accept numpy arrays
                model_name='VGG-Face',
                enforce_detection=False
            )
            
            # Extract the embedding array
            features = np.array(embedding[0]['embedding'])
            
            # Ensure consistent output shape
            if len(features) != self.output_shape:
                if len(features) < self.output_shape:
                    features = np.pad(features, (0, self.output_shape - len(features)))
                else:
                    features = features[:self.output_shape]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from array: {str(e)}")
            # Fallback to temp file method
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_array_{int(time.time())}.jpg")
            try:
                cv2.imwrite(temp_path, image_array)
                return self.extract_features(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Compare two face encodings using cosine similarity
        Returns similarity score between 0 and 1
        """
        try:
            # Ensure inputs are numpy arrays
            enc1 = np.array(encoding1, dtype=np.float32)
            enc2 = np.array(encoding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(enc1, enc2)
            norm1 = np.linalg.norm(enc1)
            norm2 = np.linalg.norm(enc2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            print(f"Error comparing faces: {str(e)}")
            return 0.0
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces in a video frame using YuNet or cascade fallback"""
        faces = []
        
        if frame is None or frame.size == 0:
            return faces
            
        try:
            height, width = frame.shape[:2]
            
            # Try YuNet first
            if self.yunet_detector is not None:
                try:
                    # Set input size
                    self.yunet_detector.setInputSize((width, height))
                    
                    # Detect faces
                    _, faces_data = self.yunet_detector.detect(frame)
                    
                    if faces_data is not None:
                        for face in faces_data:
                            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                            confidence = float(face[14]) if len(face) > 14 else 0.8
                            
                            # Ensure coordinates are within frame
                            x = max(0, min(x, width - 1))
                            y = max(0, min(y, height - 1))
                            w = max(1, min(w, width - x))
                            h = max(1, min(h, height - y))
                            
                            faces.append(((x, y, w, h), confidence))
                    
                    return faces
                    
                except Exception as e:
                    print(f"YuNet detection error: {e}")
            
            # Fallback to Haar cascade
            if self.cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                detected_faces = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in detected_faces:
                    faces.append(((int(x), int(y), int(w), int(h)), 0.8))
            
            return faces
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return []
    
    def detect_faces_in_image(self, image_path: str) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces in an image file"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return []
            
            return self.detect_faces_in_frame(frame)
            
        except Exception as e:
            print(f"Error detecting faces in image: {str(e)}")
            return []
    
    def detect_people_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect people in a video frame using HOG detector"""
        people = []
        
        if frame is None or frame.size == 0 or self.hog_detector is None:
            return people
            
        try:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Detect people
            boxes, weights = self.hog_detector.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05
            )
            
            for i, (x, y, w, h) in enumerate(boxes):
                confidence = float(weights[i]) if i < len(weights) else 0.5
                people.append(((int(x), int(y), int(w), int(h)), confidence))
            
            return people
            
        except Exception as e:
            print(f"People detection error: {str(e)}")
            return []
