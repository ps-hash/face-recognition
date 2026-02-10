import cv2
import numpy as np
import os
from datetime import datetime

# Initialize face detector with adjusted parameters
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error: Could not load face cascade classifier")
except Exception as e:
    print(f"Error loading face detector: {e}")
    exit(1)

# Initialize webcam
try:
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit(1)

# Load known faces
known_faces = {}
path = "known faces"
try:
    if not os.path.exists(path):
        raise Exception(f"Directory '{path}' does not exist")
    
    files = os.listdir(path)
    if not files:
        raise Exception(f"No images found in '{path}' directory")
    
    for img_name in files:
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            try:
                name = os.path.splitext(img_name)[0]
                img_path = os.path.join(path, img_name)
                print(f"Loading image: {img_path}")
                
                # Read and store the image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {img_name}")
                    continue
                
                # Detect faces in the known face image
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                if len(faces) > 0:
                    # Get the largest face in the image
                    x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                    face = img[y:y+h, x:x+w]
                    # Normalize the face image
                    face = cv2.equalizeHist(face)
                    known_faces[name] = face
                    print(f"Successfully loaded face for: {name}")
                else:
                    print(f"Warning: No face detected in {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue
except Exception as e:
    print(f"Error loading known faces: {e}")
    exit(1)

if not known_faces:
    print("Error: No valid face images could be loaded")
    exit(1)

print(f"Successfully loaded {len(known_faces)} known faces")

def mark_attendance(name):
    try:
        if name not in students_marked:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            date_string = now.strftime('%Y-%m-%d')
            with open("attendance.csv", "a") as f:
                f.write(f"{name},{date_string},{time_string}\n")
            students_marked.append(name)
            return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
    return False

def get_face_similarity(face1, face2):
    try:
        # Standardize size
        size = (100, 100)
        face1 = cv2.resize(face1, size)
        face2 = cv2.resize(face2, size)
        
        # Apply histogram equalization for better contrast
        face1 = cv2.equalizeHist(face1)
        
        # Calculate normalized cross-correlation
        correlation = cv2.matchTemplate(face1, face2, cv2.TM_CCORR_NORMED)[0][0]
        
        # Calculate structural similarity
        mean1, mean2 = np.mean(face1), np.mean(face2)
        std1, std2 = np.std(face1), np.std(face2)
        covariance = np.mean((face1 - mean1) * (face2 - mean2))
        
        # Avoid division by zero
        denominator = (mean1**2 + mean2**2) * (std1**2 + std2**2)
        if denominator == 0:
            ssim_score = 0
        else:
            ssim_score = (2 * mean1 * mean2) * (2 * covariance) / denominator
        
        # Combine scores with weights
        return (0.7 * correlation + 0.3 * max(0, ssim_score))
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

print("\nStarting camera... Press 'q' to quit.")
print("System will automatically stop after recognizing a known person.")

students_marked = []
person_recognized = False
consecutive_matches = 0
required_matches = 2
last_matched_name = None

try:
    while not person_recognized:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more lenient parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Find best matching face
            best_match = None
            best_similarity = 0.65  # Minimum threshold
            
            for name, known_face in known_faces.items():
                try:
                    similarity = get_face_similarity(face_roi, known_face)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                except Exception as e:
                    print(f"Error comparing with {name}: {e}")
                    continue
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # If we found a match
            if best_match:
                if best_match == last_matched_name:
                    consecutive_matches += 1
                    match_text = f"{best_match} ({consecutive_matches}/{required_matches})"
                else:
                    consecutive_matches = 1
                    last_matched_name = best_match
                    match_text = f"{best_match} (1/{required_matches})"
                
                # Draw name above face
                cv2.putText(frame, match_text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          (255, 255, 255), 2)
                
                if consecutive_matches >= required_matches:
                    # Show the final frame with recognition for 2 seconds
                    cv2.imshow('Face Attendance', frame)
                    cv2.waitKey(2000)
                    
                    # Mark attendance and set flag to stop
                    if mark_attendance(best_match):
                        print(f"Recognized {best_match}! Attendance marked.")
                        person_recognized = True
                        break
            else:
                # Reset consecutive matches if no match found
                consecutive_matches = 0
                last_matched_name = None
                cv2.putText(frame, "Unknown", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          (255, 255, 255), 2)

        if not person_recognized:
            cv2.imshow('Face Attendance', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit the program")
                break

except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("System stopped.")
