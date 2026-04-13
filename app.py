from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import json
import os
import threading
from datetime import datetime, timedelta

from utils.face_utils import FaceDetector, FaceRecognizer, AttendanceManager

app = Flask(__name__)
CORS(app)

detector = FaceDetector()
recognizer = FaceRecognizer()
attendance_manager = AttendanceManager()

camera = None
is_scanning = False
current_status = {
    "detected": False,
    "recognized": None,
    "confidence": 0
}
status_lock = threading.Lock()
capture_request = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    global is_scanning, current_status
    cam = get_camera()
    while True:
        if not is_scanning:
            # If scanning stops, wait a bit then break to close connection
            break
            
        success, frame = cam.read()
        if not success:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We need a copy of the frame to draw on it
        output_frame = frame.copy()
        faces = detector.detect_faces(gray)
        
        status_update = {
            "detected": False,
            "recognized": None,
            "confidence": 0
        }
        
        for (x, y, w, h) in faces:
            global capture_request
            face_roi = gray[y:y+h, x:x+w]
            
            if capture_request is not None and capture_request.get("status") == "pending":
                try:
                    success = recognizer.add_face(capture_request["name"], face_roi)
                    if success:
                        capture_request["status"] = "success"
                    else:
                        capture_request["status"] = "error"
                        capture_request["error"] = "Failed to add face"
                except Exception as e:
                    capture_request["status"] = "error"
                    capture_request["error"] = str(e)
                # Consume this face for capture, do not process regular attendance
                break
                
            status_update["detected"] = True
            
            try:
                name, confidence = recognizer.recognize_face(face_roi)
                conf_pct = int(confidence * 100)
                status_update["confidence"] = conf_pct
                
                if name is not None and confidence > 0.65:
                    status_update["recognized"] = name
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{name} ({conf_pct}%)"
                    cv2.putText(output_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if confidence > 0.8:
                        attendance_manager.mark_attendance(name)
                else:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(output_frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Recognition error: {e}")
                
            # Process largest face
            break
            
        with status_lock:
            current_status = status_update
            
        ret, buffer = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/api/video_feed')
def video_feed():
    global is_scanning
    is_scanning = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global is_scanning
    is_scanning = False
    release_camera()
    return jsonify({"status": "stopped"})

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global is_scanning
    is_scanning = True
    return jsonify({"status": "started"})

import time

@app.route('/api/add_face', methods=['POST'])
def add_face():
    global capture_request
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({"success": False, "error": "Name required"}), 400
        
    # Strip spaces for filename safety
    safe_name = "".join(x for x in name if x.isalnum() or x in "._- ")
    
    capture_request = {"name": safe_name, "status": "pending"}
    
    # Wait for up to 5 seconds
    t0 = time.time()
    while time.time() - t0 < 5:
        if capture_request["status"] == "success":
            capture_request = None
            return jsonify({"success": True})
        elif capture_request["status"] == "error":
            err = capture_request.get("error", "Failed")
            capture_request = None
            return jsonify({"success": False, "error": err}), 400
        time.sleep(0.1)
        
    capture_request = None
    return jsonify({"success": False, "error": "No face detected in camera within 5 seconds."}), 400

@app.route('/api/scan_status', methods=['GET'])
def scan_status():
    with status_lock:
        return jsonify(current_status)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Read present count today
    today = datetime.now().strftime('%Y-%m-%d')
    present_set = set()
    total_employees = len(recognizer.known_faces)
    
    if os.path.exists(attendance_manager.csv_path):
        with open(attendance_manager.csv_path, 'r') as f:
            for line in f:
                if line.strip():
                    name, date_str, time_str = line.strip().split(',')
                    if date_str == today:
                        present_set.add(name)
                        
    present = len(present_set)
    absent = total_employees - present
    if total_employees > 0:
        pct = (present / total_employees) * 100
    else:
        pct = 0
        
    return jsonify({
        "total": total_employees,
        "present": present,
        "absent": absent,
        "percentage": f"{pct:.1f}%"
    })

@app.route('/api/attendance/recent', methods=['GET'])
def get_recent():
    recent = []
    if os.path.exists(attendance_manager.csv_path):
        with open(attendance_manager.csv_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in reversed(lines[-20:]): # top 20 recent
                name, date_str, time_str = line.split(',')
                recent.append({
                    "name": name,
                    "time": time_str,
                    "status": "success"
                })
    return jsonify(recent)

@app.route('/api/attendance/weekly', methods=['GET'])
def get_weekly():
    # Last 7 days
    days = []
    data = []
    for i in range(6, -1, -1):
        d = datetime.now() - timedelta(days=i)
        days.append(d.strftime('%Y-%m-%d'))
        data.append({
            "day": d.strftime('%a'),
            "date": d.strftime('%Y-%m-%d'),
            "present": 0,
            "absent": 0,
            "late": 0 # Not tracked
        })
        
    date_to_index = {d['date']: i for i, d in enumerate(data)}
    
    attendance_by_day = {d: set() for d in days}
    
    if os.path.exists(attendance_manager.csv_path):
        with open(attendance_manager.csv_path, 'r') as f:
            for line in f:
                if line.strip():
                    name, date_str, time_str = line.strip().split(',')
                    if date_str in attendance_by_day:
                        attendance_by_day[date_str].add(name)
                        
    total = len(recognizer.known_faces)
    for i, d in enumerate(days):
        present = len(attendance_by_day[d])
        data[i]["present"] = present
        data[i]["absent"] = total - present
        
    return jsonify(data)

@app.route('/')
def index():
    html_content = """
    <html>
        <head><title>Backend Running</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1 style="color: #2563eb;">API Backend is Successfully Running!</h1>
            <p style="font-size: 18px; color: #4b5563;">You have accessed the Python Server which powers the Face Detection logic.</p>
            <div style="background-color: #f3f4f6; padding: 20px; border-radius: 8px; max-width: 600px; margin: 30px auto; text-align: left;">
                <h3 style="margin-top: 0; color: #1f2937;">How to see the User Interface:</h3>
                <p>The UI is a separate React application located in the <code>ui</code> folder. To view the dashboard, you must run it natively:</p>
                <ol>
                    <li>Open a new terminal window.</li>
                    <li>Navigate to the UI folder: <code>cd "ui"</code></li>
                    <li>Install dependencies: <code>npm install</code></li>
                    <li>Start the app: <code>npm run dev</code></li>
                </ol>
                <p>Then, click the <b>http://localhost:5173</b> link provided in that new terminal!</p>
            </div>
        </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
