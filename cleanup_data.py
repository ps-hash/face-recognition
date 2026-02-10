import os
import shutil

print("Starting cleanup...")

# Clear known faces
faces_dir = "known faces"
if os.path.exists(faces_dir):
    for filename in os.listdir(faces_dir):
        file_path = os.path.join(faces_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    print(f"Directory {faces_dir} does not exist.")

# Clear attendance CSV
csv_file = "attendance.csv"
if os.path.exists(csv_file):
    try:
        open(csv_file, 'w').close()
        print(f"Cleared {csv_file}")
    except Exception as e:
        print(f"Failed to clear {csv_file}. Reason: {e}")
else:
    print(f"File {csv_file} does not exist.")

print("Cleanup complete.")
