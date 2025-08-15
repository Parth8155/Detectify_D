"""
Face Recognition Client using DeepFace
Implements the FacialRecognition interface as specified in the project requirements
"""

import numpy as np
import cv2
import os
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


class DeepFaceClient(FacialRecognition):
    """
    DeepFace implementation for face recognition
    Uses DeepFace model with 4096-dimensional feature extraction
    """
    
    def __init__(self):
        self.model_name = "DeepFace"
        self.input_shape = (152, 152)
        self.output_shape = 4096
        self._initialize_model()
        self._initialize_yunet()
    
    def _initialize_model(self):
        """Initialize the DeepFace model"""
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
        """Initialize YuNet face detector"""
        try:
            # Download YuNet model if not exists
            model_path = os.path.join(tempfile.gettempdir(), 'face_detection_yunet_2023mar.onnx')
            if not os.path.exists(model_path):
                print("Downloading YuNet model...")
                import urllib.request
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(url, model_path)
                print("YuNet model downloaded successfully")
            
            # Initialize YuNet detector
            self.yunet = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (320, 240),  # Default input size
                0.9,  # Score threshold
                0.3,  # NMS threshold
                5000  # Top K
            )
            print("YuNet face detector initialized successfully")
        except Exception as e:
            print(f"Error initializing YuNet: {str(e)}")
            # Fallback to opencv cascade
            self.yunet = None
            try:
                self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Fallback: Using Haar cascade detector")
            except Exception as cascade_e:
                print(f"Error loading cascade: {str(cascade_e)}")
                self.cascade = None
        
        # Initialize HOG people detector
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("HOG people detector initialized successfully")
        except Exception as e:
            print(f"Error initializing HOG people detector: {str(e)}")
            self.hog = None
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract facial features from an image using DeepFace
        First detects faces using YuNet, then crops the largest face for feature extraction.
        """
        try:
            # Use cross-platform temp directory if needed
            temp_dir = tempfile.gettempdir()
            if image_path.startswith("/tmp/"):
                # Convert to Windows temp path if needed
                image_path = os.path.join(temp_dir, os.path.basename(image_path))
            if not os.path.exists(image_path):
                print(f"File does not exist: {image_path}")
                return np.zeros(self.output_shape)
            
            # First detect faces in the image
            detected_faces = self.detect_faces_in_image(image_path)
            
            if detected_faces:
                # Use the largest detected face
                largest_face = max(detected_faces, key=lambda x: x[0][2] * x[0][3])
                x, y, w, h = largest_face[0]
                
                # Load and crop the image to the detected face
                image = cv2.imread(image_path)
                if image is not None:
                    face_crop = image[y:y+h, x:x+w]
                    
                    # Save cropped face temporarily
                    temp_face_path = os.path.join(temp_dir, f"temp_cropped_face_{os.path.basename(image_path)}")
                    cv2.imwrite(temp_face_path, face_crop)
                    
                    try:
                        # Extract features from cropped face
                        embedding = self.deepface.represent(
                            img_path=temp_face_path,
                            model_name='Facenet512',
                            enforce_detection=False  # We already detected and cropped
                        )
                        
                        # Convert to numpy array and ensure correct shape
                        features = np.array(embedding[0]['embedding'])
                        
                        # Pad or truncate to match output_shape if necessary
                        if len(features) != self.output_shape:
                            if len(features) < self.output_shape:
                                features = np.pad(features, (0, self.output_shape - len(features)))
                            else:
                                features = features[:self.output_shape]
                        
                        return features
                    finally:
                        print()
                        # # Clean up temp file
                        # if os.path.exists(temp_face_path):
                        #     os.remove(temp_face_path)
            
            # Fallback: use original image if no faces detected
            print(f"No faces detected in {image_path}, using full image")
            embedding = self.deepface.represent(
                img_path=image_path,
                model_name='VGG-Face',
                enforce_detection=False
            )
            
            # Convert to numpy array and ensure correct shape
            features = np.array(embedding[0]['embedding'])
            
            # Pad or truncate to match output_shape if necessary
            if len(features) != self.output_shape:
                if len(features) < self.output_shape:
                    features = np.pad(features, (0, self.output_shape - len(features)))
                else:
                    features = features[:self.output_shape]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return np.zeros(self.output_shape)
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Compare two face encodings using cosine similarity
        
        Args:
            encoding1 (np.ndarray): First face encoding
            encoding2 (np.ndarray): Second face encoding
            
        Returns:
            float: Similarity score between 0 and 1 (1 = identical)
        """
        try:
            # Calculate cosine similarity
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            
            # Convert to similarity score (0-1 range)
            similarity = (cosine_similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error comparing face encodings: {str(e)}")
            return 0.0
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in a video frame using YuNet detector
        
        Args:
            frame (np.ndarray): Video frame as numpy array
            
        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: List of (bounding_box, confidence) tuples
        """
        try:
            results = []
            h, w = frame.shape[:2]
            
            if hasattr(self, 'yunet') and self.yunet is not None:
                # Use YuNet detector
                self.yunet.setInputSize((w, h))
                _, faces = self.yunet.detect(frame)
                
                if faces is not None:
                    for face in faces:
                        x, y, w_face, h_face = face[:4].astype(int)
                        confidence = face[14] if len(face) > 14 else 0.9
                        
                        # Ensure bounding box is within frame
                        x = max(0, x)
                        y = max(0, y)
                        w_face = min(w_face, w - x)
                        h_face = min(h_face, h - y)
                        
                        if w_face > 20 and h_face > 20:  # Minimum face size
                            results.append(((x, y, w_face, h_face), float(confidence)))
                            
            elif hasattr(self, 'cascade') and self.cascade is not None:
                # Fallback to Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w_face, h_face) in faces:
                    if w_face > 20 and h_face > 20:
                        results.append(((x, y, w_face, h_face), 0.8))
            else:
                print("No face detector available")
                
            return results
            
        except Exception as e:
            print(f"Error detecting faces in frame: {str(e)}")
            return []
    
    def detect_faces_in_image(self, image_path: str) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in an image using YuNet detector
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: List of (bounding_box, confidence) tuples
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return []
            
            return self.detect_faces_in_frame(image)
            
        except Exception as e:
            print(f"Error detecting faces in image {image_path}: {str(e)}")
            return []
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for face recognition
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to input shape
            image = cv2.resize(image, self.input_shape)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def detect_people_in_frame(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect people in a video frame using HOG descriptor
        
        Args:
            frame (np.ndarray): Video frame as numpy array
            
        Returns:
            List[Tuple[Tuple[int, int, int, int], float]]: List of (bounding_box, confidence) tuples
        """
        try:
            results = []
            
            if hasattr(self, 'hog') and self.hog is not None:
                # Resize frame for better detection
                height, width = frame.shape[:2]
                scale = 1.0
                if height > 600:
                    scale = 600 / height
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    resized_frame = frame
                
                # Detect people using HOG (remove unsupported arguments for OpenCV version compatibility)
                (people, weights) = self.hog.detectMultiScale(
                    resized_frame,
                    winStride=(4, 4),
                    padding=(8, 8),
                    scale=1.05
                )
                
                # Scale back coordinates if frame was resized
                for i, (x, y, w, h) in enumerate(people):
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)
                    
                    # Ensure bounding box is within frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 50 and h > 100:  # Minimum person size
                        confidence = weights[i] if i < len(weights) else 0.8
                        results.append(((x, y, w, h), float(confidence)))
            else:
                print("HOG people detector not available")
                
            return results
            
        except Exception as e:
            print(f"Error detecting people in frame: {str(e)}")
            return []


# Factory function to create DeepFace client
def create_face_recognition_client() -> DeepFaceClient:
    """Factory function to create a DeepFace client instance"""
    return DeepFaceClient()
