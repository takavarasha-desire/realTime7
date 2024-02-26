"""
import os
import random
import shutil

# Define the paths
root_folder = "C:\\Users\\takav\PycharmProjects\\realTime\\extracted_data\\FERG_DB_256"
output_folder = 'data_split'
train_ratio = 0.8  # 80% for train, 20% for test

# Create the output folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)

# Collect all image paths
image_paths = []
for folder in os.listdir(root_folder):
    person_folder = os.path.join(root_folder, folder)
    if not os.path.isdir(person_folder):
        continue  # Skip non-directory items
    for emotion_folder in os.listdir(person_folder):
        emotion_folder_path = os.path.join(person_folder, emotion_folder)
        if not os.path.isdir(emotion_folder_path):
            continue  # Skip non-directory items
        image_paths += [os.path.join(emotion_folder_path, filename) for filename in os.listdir(emotion_folder_path)]

# Shuffle the image paths
random.shuffle(image_paths)

# Split the image paths into train and test sets
split_index = int(train_ratio * len(image_paths))
train_image_paths = image_paths[:split_index]
test_image_paths = image_paths[split_index:]

# Copy images to train and test folders
for image_path in train_image_paths:
    shutil.copy(image_path, os.path.join(output_folder, 'train'))

for image_path in test_image_paths:
    shutil.copy(image_path, os.path.join(output_folder, 'test'))

"""
#####################################################################################################################
import os
import shutil
import glob
from PIL import Image
import PIL

# Function to get the class label from the filename
def get_class_label(filename):
    class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    for class_name in class_names:
        if class_name in filename:
            return class_name
    return None

# Define the source directory containing all the images
source_directory = 'path_to_directory_containing_images'

# Define the destination directory where the images will be sorted into subfolders
destination_directory = 'path_to_destination_directory'

# Create subdirectories for each class in the destination directory
class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
for class_name in class_names:
    class_directory = os.path.join(destination_directory, class_name)
    if not os.path.exists(class_directory):
        os.makedirs(class_directory)

# Get the list of image files in the source directory
image_files = glob.glob(os.path.join(source_directory, '*.png'))

# Move each image to the appropriate subdirectory based on its class label
for image_file in image_files:
    filename = os.path.basename(image_file)
    class_label = get_class_label(filename)
    if class_label is not None:
        try:
            # Open the image using PIL to check if it is a valid image
            img = Image.open(image_file)
            destination_path = os.path.join(destination_directory, class_label, filename)
            shutil.move(image_file, destination_path)
        except (IOError, OSError, PIL.UnidentifiedImageError):
            print(f"Skipped invalid image: {filename}")
            # You can choose to delete or handle the invalid image differently, based on your requirements.
