import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from utils.face_utils import FaceDetector, FaceRecognizer, AttendanceManager
import time

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x700")
        
        # Initialize utilities
        try:
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer()
            self.attendance_manager = AttendanceManager()
            
            # Initialize video capture
            print("Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            messagebox.showerror("Initialization Error", str(e))
            self.cap = None
            
        self.is_capturing = False
        self._create_gui()
        
    def _create_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed and controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = ttk.Label(left_panel)
        self.video_label.pack(pady=10)
        
        # Control buttons
        controls = ttk.Frame(left_panel)
        controls.pack(pady=10)
         
        ttk.Button(controls, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Add Face", command=self.add_face).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Attendance list
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Attendance list
        ttk.Label(right_panel, text="Attendance List", font=('Helvetica', 12, 'bold')).pack(pady=5)
        
        # Treeview for attendance
        self.tree = ttk.Treeview(right_panel, columns=('Name', 'Time'), show='headings')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Time', text='Time')
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Statistics
        self.stats_label = ttk.Label(right_panel, text="Present: 0 | Total: 0 | 0%")
        self.stats_label.pack(pady=5)
        
        # Delete button
        ttk.Button(right_panel, text="Delete Selected", command=self.delete_attendance).pack(pady=5)
        
        # Reset button
        ttk.Button(right_panel, text="Reset All Data", command=self.reset_data).pack(pady=5)
        
    def reset_data(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to delete ALL facial data and attendance records? This cannot be undone."):
            try:
                # Clear known faces
                faces_dir = "known faces"
                if os.path.exists(faces_dir):
                    import shutil
                    shutil.rmtree(faces_dir)
                    os.makedirs(faces_dir)
                
                # Clear attendance CSV
                open(self.attendance_manager.csv_path, 'w').close()
                
                # Clear memory
                self.face_recognizer.known_faces.clear()
                self.attendance_manager.marked_students.clear()
                
                # Refresh UI
                self.update_attendance_list()
                self.stats_label.configure(text="Present: 0 | Total: 0 | 0%")
                
                messagebox.showinfo("Success", "All data has been reset successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset data: {e}")

    def delete_attendance(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a record to delete")
            return
            
        item = self.tree.item(selected_item)
        values = item['values']
        name = values[0]
        date_time = values[1]
        
        if messagebox.askyesno("Confirm", f"Delete attendance for {name} at {date_time}?"):
            if self.attendance_manager.delete_record(name, date_time):
                self.update_attendance_list()
                messagebox.showinfo("Success", "Record deleted successfully")
            else:
                messagebox.showerror("Error", "Failed to delete record")
        
    def start_camera(self):
        """Start the camera capture."""
        try:
            if self.cap is None:
                # Try to initialize camera again if it failed before
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open webcam")
            
            # Try to read a test frame
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame from camera")
            
            self.is_capturing = True
            self.update_frame()
            print("Camera started successfully")
            
        except Exception as e:
            error_msg = f"Failed to start camera: {str(e)}"
            print(error_msg)
            messagebox.showerror("Camera Error", error_msg)
            self.is_capturing = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def stop_camera(self):
        """Stop the camera capture."""
        print("Stopping camera...")
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.video_label.configure(image='')
        print("Camera stopped")
    
    def update_frame(self):
        if self.is_capturing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with more lenient parameters
                faces = self.face_detector.detect_faces(frame)
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    try:
                        name, confidence = self.face_recognizer.recognize_face(face_roi)
                        
                        # Draw rectangle around face
                        if name is not None and confidence > 0.65:
                            # Draw green rectangle for recognized face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            label = f"{name} ({confidence:.2f})"
                            
                            # Mark attendance if confidence is high enough
                            if confidence > 0.8:
                                if self.attendance_manager.mark_attendance(name):
                                    self.update_attendance_list()
                                    # Show recognition message
                                    self.root.after(0, lambda n=name: 
                                        messagebox.showinfo("Recognition", 
                                            f"Recognized {n}! Attendance marked."))
                        else:
                            # Draw red rectangle for unrecognized face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            label = "Unknown"
                        
                        # Draw name above face
                        cv2.putText(frame, label, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                  (255, 255, 255), 2)
                            
                    except Exception as e:
                        print(f"Error in face recognition: {e}")
                        # Draw red rectangle for error
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Error", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                photo = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            
            if self.is_capturing:
                self.root.after(10, self.update_frame)
    
    def add_face(self):
        if not self.is_capturing:
            messagebox.showerror("Error", "Please start the camera first")
            return
            
        try:
            print("Attempting to capture frame...")
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not capture frame from camera")
                print("Failed to capture frame")
                return
            print("Frame captured successfully")
            
            # Convert to grayscale first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print("Detecting faces...")
            faces = self.face_detector.detect_faces(gray)  # Pass grayscale image directly
            print(f"Found {len(faces)} faces")
            
            if len(faces) == 0:
                messagebox.showerror("Error", "No face detected. Please ensure your face is visible in the frame.")
                return
                
            if len(faces) > 1:
                messagebox.showerror("Error", "Multiple faces detected. Please ensure only one person is in frame.")
                return
            
            # Get the largest face
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            
            # Apply histogram equalization
            face_img = cv2.equalizeHist(face_img)
            
            # Get name from user
            name = simpledialog.askstring("Input", "Enter person's name:")
            if not name:
                print("No name provided")
                return
            print(f"Name provided: {name}")
            
            # Ensure directory exists
            if not os.path.exists("known faces"):
                os.makedirs("known faces")
            
            # Save the face image
            img_path = os.path.join("known faces", f"{name}.png")
            if cv2.imwrite(img_path, face_img):
                # Add to recognizer's known faces
                self.face_recognizer.known_faces[name] = face_img
                print(f"Successfully added face for: {name}")
                messagebox.showinfo("Success", f"Face successfully added for {name}")
                
                # Show preview
                preview = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                preview = cv2.resize(preview, (200, 200))
                preview_img = Image.fromarray(preview)
                preview_photo = ImageTk.PhotoImage(image=preview_img)
                
                preview_window = tk.Toplevel(self.root)
                preview_window.title("Face Preview")
                preview_label = ttk.Label(preview_window, image=preview_photo)
                preview_label.image = preview_photo
                preview_label.pack(padx=10, pady=10)
                ttk.Label(preview_window, text=f"Saved face for {name}").pack(pady=5)
            else:
                raise Exception("Failed to save face image")
                
        except Exception as e:
            print(f"Error in add_face: {e}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def update_attendance_list(self):
        # Clear current items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add attendance records
        with open(self.attendance_manager.csv_path, 'r') as f:
            for line in f:
                name, date, time = line.strip().split(',')
                self.tree.insert('', 'end', values=(name, f"{date} {time}"))
        
        # Update statistics
        present, total, percentage = self.attendance_manager.get_attendance_stats()
        self.stats_label.configure(text=f"Present: {present} | Total: {total} | {percentage:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop() 