import subprocess
import time
import os
import sys

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# Ensure no console windows pop up
CREATE_NO_WINDOW = 0x08000000

try:
    # 1. Start the Flask Backend
    # Using 'py' as it was confirmed to work on this machine.
    backend_proc = subprocess.Popen(
        ["py", "app.py"], 
        cwd=base_dir, 
        creationflags=CREATE_NO_WINDOW
    )

    # 2. Start the Vite Frontend
    ui_dir = os.path.join(base_dir, "ui")
    frontend_proc = subprocess.Popen(
        "npm run dev", 
        shell=True, 
        cwd=ui_dir, 
        creationflags=CREATE_NO_WINDOW
    )

    # Give the servers a moment to initialize
    time.sleep(3)

    # 3. Launch the Application Window
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_path = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_path = p
            break

    if edge_path:
        # Run Edge in app mode, this blocks until the user closes the window
        subprocess.run(
            [edge_path, "--app=http://localhost:5173", "--new-window"]
        )
    else:
        # Fallback if edge is not found
        subprocess.run(
            "start http://localhost:5173", 
            shell=True
        )
finally:
    # 4. Clean-up servers when the window is closed
    if 'backend_proc' in locals():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(backend_proc.pid)], creationflags=CREATE_NO_WINDOW)
    
    if 'frontend_proc' in locals():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], creationflags=CREATE_NO_WINDOW)
