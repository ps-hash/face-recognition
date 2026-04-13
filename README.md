# FaceTrack AI: Smart Attendance System

A modern, full-stack web application for automated facial recognition and attendance tracking. The system utilizes a robust Python OpenCV backend wrapped in a Flask REST API, paired with a beautiful, fully responsive React & TailwindCSS dashboard.

## 🌟 Key Features

- **Real-Time Video Streaming**: Low-latency MJPEG video feed streamed directly from Python to the React UI.
- **Dynamic Face Recognition**: OpenCV Haar Cascades paired with custom similarity-based face matching algorithms.
- **In-Browser Employee Registration**: Capture and register new employee faces directly from the web dashboard.
- **Live Analytical Dashboard**: Instantly tracks attendance records via an embedded CSV database and visualizes daily presence percentages.
- **One-Click Native Launcher**: Completely skips the terminal using `Start FaceTrack.bat` to boot the full application suite seamlessly.

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask, OpenCV, NumPy
- **Frontend**: React 18, Vite, TailwindCSS v4, Recharts, Lucide Icons
- **Database**: Local File System (`known faces/`) & CSV Logging (`attendance.csv`)

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/udit-jhanjhariya/Smart-Attendance-System.git
cd Smart-Attendance-System
```

2. **Install Backend Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Frontend Dependencies:**
Ensure you have [Node.js](https://nodejs.org) installed on your system.
```bash
cd ui
npm install
```

## 💻 Usage

### The Easiest Way (Windows)
Simply double click the **`Start FaceTrack.bat`** file in the root of the project! 
This batch script autonomously launches the Python recognition engine, compiles the Node dashboard, and spawns the UI in a native-looking App Container.

### The Manual Developer Way
If you wish to modify code and develop manually:

**1. Start the Python API Server:**
```bash
python app.py
```
*(The backend will run on `http://127.0.0.1:5000`)*

**2. Start the React UI Server:**
Open a new terminal window:
```bash
cd ui
npm run dev
```
*(Navigate to `http://localhost:5173` in your web browser)*

## 📁 Architecture Structure

```text
smart-attendance-system/
│
├── Start FaceTrack.bat       # Supervised one-click launcher
├── app.py                    # Flask API & MJPEG camera stream logic
├── utils/
│   └── face_utils.py         # OpenCV detector & recognizer implementation
├── ui/                       # React + Vite Frontend Workspace
│   ├── index.html            # Vite entry point
│   ├── src/                  # React Components (Dashboard, FaceScanner, etc)
│   └── styles/               # Tailwind definitions and CSS variables
├── known faces/              # Encoded JPG/PNG facial databases
├── attendance.csv            # Parsed daily attendance logs
└── requirements.txt          # Python dependencies
```

## ⚙️ Configuration
- The default confidence recognition threshold is `0.65` in `FaceRecognizer`.
- Bounding-box detection frame updates asynchronously for peak performance.
- The React Development proxy forwards all `/api` calls safely through port `5000` avoiding CORS issues.