import os
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


# Defining the source directories for train and test data
train_directory = 'C:/Users/takav/PycharmProjects/realTime/data_split/train'
test_directory = 'C:/Users/takav/PycharmProjects/realTime/data_split/test'


# Function to remove unidentified images in a directory
def remove_unidentified_images(directory):
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for image_file in glob.glob(os.path.join(class_dir, '*.png')):
                try:
                    # Open the image using PIL to check if it is a valid image
                    img = Image.open(image_file)
                except (IOError, OSError, PIL.UnidentifiedImageError):
                    print(f"Removed invalid image: {image_file}")
                    os.remove(image_file)


# Remove unidentified images from the train and test directories
remove_unidentified_images(train_directory)
remove_unidentified_images(test_directory)
