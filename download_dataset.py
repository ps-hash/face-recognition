import os
import shutil
import kagglehub

TARGET_DIR = "known faces"
NUM_FACES_TO_EXTRACT = 1000

def fetch_and_extract_faces():
    print("Downloading LFW dataset from Kaggle via kagglehub...")
    print("This may take a few minutes...")
    
    try:
        # Download latest version of the LFW dataset
        dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Determine where the faces are located within the dataset
        # Usually it's in lfw-deepfunneled or lfw/lfw folder
        faces_source_dir = None
        
        # Traverse the dataset directory to find images
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
                    
        print(f"Found {len(image_files)} face images overall in the dataset.")
        
        if len(image_files) == 0:
            print("Could not find any face images in the dataset.")
            return False

        # Create target directory
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)
            
        print(f"Extracting up to {NUM_FACES_TO_EXTRACT} images to '{TARGET_DIR}'...")
        
        extracted_count = 0
        for img_path in image_files:
            if extracted_count >= NUM_FACES_TO_EXTRACT:
                break
                
            # Use original filename
            basename = os.path.basename(img_path)
            dest_path = os.path.join(TARGET_DIR, basename)
            
            # Avoid overwriting or keeping duplicate files (if any have same name)
            # In LFW they usually have unique names like Aaron_Eckhart_0001.jpg
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                extracted_count += 1
                
            if extracted_count > 0 and extracted_count % 100 == 0:
                print(f"Extracted... {extracted_count}/{NUM_FACES_TO_EXTRACT}")
                
        print(f"Successfully copied {extracted_count} face images.")
        return True
        
    except Exception as e:
        print(f"Error accessing or extracting dataset: {e}")
        return False

if __name__ == '__main__':
    fetch_and_extract_faces()
    print("Done!")
