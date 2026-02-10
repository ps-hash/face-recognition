import cv2
import numpy as np
import os
from typing import Tuple, List, Dict, Optional

class FaceDetector:
    """Face detection and recognition utility class."""
    
    def __init__(self, cascade_path: str = None):
        """Initialize the face detector.
        
        Args:
            cascade_path (str, optional): Path to cascade classifier XML file.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Error: Could not load face cascade classifier")
    
    def detect_faces(self, image: np.ndarray, 
                    scale_factor: float = 1.1, 
                    min_neighbors: int = 4,
                    min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image
            scale_factor (float): Detection scale factor
            min_neighbors (int): Minimum neighbors for detection
            min_size (tuple): Minimum face size to detect
            
        Returns:
            List[Tuple]: List of face coordinates (x, y, w, h)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        return self.face_cascade.detectMultiScale(
            gray, scale_factor, min_neighbors, minSize=min_size
        )

class FaceRecognizer:
    """Face recognition utility class."""
    
    def __init__(self, threshold: float = 0.65):
        """Initialize the face recognizer.
        
        Args:
            threshold (float): Similarity threshold for recognition
        """
        self.threshold = threshold
        self.known_faces: Dict[str, np.ndarray] = {}
        self.known_faces_dir = "known faces"
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            
        # Load existing known faces
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known faces from disk."""
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.known_faces_dir, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Preprocess the loaded face
                        img = self._preprocess_face(img)
                        self.known_faces[name] = img
                        print(f"Loaded face for {name}")
                except Exception as e:
                    print(f"Error loading face for {name}: {e}")
    
    def add_face(self, name: str, face_image: np.ndarray) -> bool:
        """Add a face to the known faces database.
        
        Args:
            name (str): Name of the person
            face_image (np.ndarray): Face image
            
        Returns:
            bool: True if face was added successfully
        """
        try:
            print(f"Adding face for {name}")
            print(f"Input face image shape: {face_image.shape}")
            
            # Preprocess the face
            face = self._preprocess_face(face_image)
            print(f"Preprocessed face shape: {face.shape}")
            
            # Save the face image
            img_path = os.path.join(self.known_faces_dir, f"{name}.png")
            print(f"Saving face to: {img_path}")
            
            # Ensure the directory exists
            os.makedirs(self.known_faces_dir, exist_ok=True)
            
            # Save the image
            success = cv2.imwrite(img_path, face)
            if not success:
                print("Failed to save face image")
                return False
            print("Face image saved successfully")
            
            # Add to memory
            self.known_faces[name] = face
            print(f"Face added to memory for {name}")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def _calculate_similarity(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """Calculate similarity between two face images.
        
        Args:
            face1 (np.ndarray): First face image
            face2 (np.ndarray): Second face image
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Standardize size
            size = (100, 100)
            face1 = cv2.resize(face1, size)
            face2 = cv2.resize(face2, size)
            
            # Apply histogram equalization for better contrast
            face1 = cv2.equalizeHist(face1)
            
            # Calculate normalized cross-correlation
            result = cv2.matchTemplate(face1, face2, cv2.TM_CCORR_NORMED)
            correlation = float(result[0][0])
            
            # Calculate structural similarity
            mean1 = float(np.mean(face1))
            mean2 = float(np.mean(face2))
            std1 = float(np.std(face1))
            std2 = float(np.std(face2))
            covariance = float(np.mean((face1 - mean1) * (face2 - mean2)))
            
            # Avoid division by zero
            denominator = float((mean1**2 + mean2**2) * (std1**2 + std2**2))
            if denominator == 0:
                ssim_score = 0.0
            else:
                ssim_score = float((2 * mean1 * mean2) * (2 * covariance) / denominator)
            
            # Combine scores with weights and ensure float return
            similarity = float(0.7 * correlation + 0.3 * max(0, ssim_score))
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face from known faces.
        
        Args:
            face_image (np.ndarray): Face image to recognize
            
        Returns:
            Tuple[str, float]: (Name of best match, similarity score)
        """
        try:
            # Preprocess the face
            face = self._preprocess_face(face_image)
            
            best_match = None
            best_similarity = float(0.65)  # Minimum threshold
            
            for name, known_face in self.known_faces.items():
                try:
                    similarity = float(self._calculate_similarity(face, known_face))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                except Exception as e:
                    print(f"Error comparing with {name}: {e}")
                    continue
            
            return best_match, float(best_similarity)
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None, 0.0
    
    def _preprocess_face(self, face: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Preprocess face image for recognition.
        
        Args:
            face (np.ndarray): Input face image
            size (tuple): Target size for face image
            
        Returns:
            np.ndarray: Preprocessed face image
        """
        try:
            # Convert to grayscale if not already
            if len(face.shape) == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            face = cv2.resize(face, size)
            
            # Apply histogram equalization
            face = cv2.equalizeHist(face)
            
            return face
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            raise

class AttendanceManager:
    """Manage attendance records."""
    
    def __init__(self, csv_path: str = "attendance.csv"):
        """Initialize attendance manager.
        
        Args:
            csv_path (str): Path to attendance CSV file
        """
        self.csv_path = csv_path
        self.marked_students = set()
        
    def mark_attendance(self, name: str) -> bool:
        """Mark attendance for a person.
        
        Args:
            name (str): Name of the person
            
        Returns:
            bool: True if attendance was marked successfully
        """
        if name in self.marked_students:
            return False
            
        try:
            from datetime import datetime
            now = datetime.now()
            with open(self.csv_path, "a") as f:
                f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
            self.marked_students.add(name)
            return True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False

    def delete_record(self, name: str, date_time: str) -> bool:
        """Delete an attendance record.
        
        Args:
            name (str): Name of the person
            date_time (str): Date and time string to match
            
        Returns:
            bool: True if record was deleted
        """
        try:
            lines = []
            deleted = False
            with open(self.csv_path, 'r') as f:
                lines = f.readlines()
            
            with open(self.csv_path, 'w') as f:
                for line in lines:
                    if not line.strip(): continue
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        record_name = parts[0]
                        record_date = parts[1]
                        record_time = parts[2]
                        record_dt = f"{record_date} {record_time}"
                        
                        if record_name == name and record_dt == date_time:
                            deleted = True
                            continue
                    f.write(line)
            
            if deleted and name in self.marked_students:
                # We can't easily know if they have other records today without re-reading
                # but for now let's just remove them from the session cache
                self.marked_students.remove(name)
                
            return deleted
        except Exception as e:
            print(f"Error deleting record: {e}")
            return False
    
    def get_attendance_stats(self) -> Tuple[int, int, float]:
        """Get attendance statistics.
        
        Returns:
            Tuple[int, int, float]: (Present count, Total count, Percentage)
        """
        present_count = len(self.marked_students)
        # In a real application, you would get total count from a database
        total_count = present_count  # Placeholder
        percentage = (present_count / total_count * 100) if total_count > 0 else 0
        return present_count, total_count, percentage 