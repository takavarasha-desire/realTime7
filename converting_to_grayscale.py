import cv2
import os

# Define the paths to the train and test folders
train_dir = "./fer2013_data/train"
test_dir = "./fer2013_data/test"

# Function to convert images in a folder to grayscale
def convert_to_grayscale(folder_path):
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        for filename in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, filename)
            if os.path.isfile(image_path) and any(image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                img_color = cv2.imread(image_path)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(image_path, img_gray)

# Create the target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Convert images in train folders to grayscale
convert_to_grayscale(train_dir)

# Convert images in test folders to grayscale
convert_to_grayscale(test_dir)
