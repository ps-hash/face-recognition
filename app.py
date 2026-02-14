from flask import Flask, render_template, request
import face_recognition
import os
import csv
from datetime import datetime

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

known_encodings = []
known_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

def mark_attendance(name):
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

@app.route("/", methods=["GET", "POST"])
def index():
    recognized = []

    if request.method == "POST":
        file = request.files["image"]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if True in matches:
                index = matches.index(True)
                name = known_names[index]
                recognized.append(name)
                mark_attendance(name)

    return render_template("index.html", recognized=recognized)

if __name__ == "__main__":
    app.run(debug=True)
