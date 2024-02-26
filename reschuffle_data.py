import os
import shutil
import random

# Defining the paths to original training and test directories
original_train_dir = 'fer2013_data/train'
original_test_dir = 'fer2013_data/test'

# Defining the paths to the new directories where reshuffled data will be stored
reshuffled_train_dir = 'fer2013_data/reshuffled_train'
reshuffled_test_dir = 'fer2013_data/reshuffled_test'

# Creating the reshuffled directories if they don't exist
os.makedirs(reshuffled_train_dir, exist_ok=True)
os.makedirs(reshuffled_test_dir, exist_ok=True)

# Listing all the class subdirectories in the original training and test directories
classes = os.listdir(original_train_dir)

# Shuffle the order of the class directories
random.shuffle(classes)

# Iterating through the class directories and moving the files to the reshuffled directories
for class_name in classes:
    original_class_train_dir = os.path.join(original_train_dir, class_name)
    original_class_test_dir = os.path.join(original_test_dir, class_name)

    reshuffled_class_train_dir = os.path.join(reshuffled_train_dir, class_name)
    reshuffled_class_test_dir = os.path.join(reshuffled_test_dir, class_name)

    # Creating the class directories in the reshuffled directories
    os.makedirs(reshuffled_class_train_dir, exist_ok=True)
    os.makedirs(reshuffled_class_test_dir, exist_ok=True)

    # Listing all files in the original class directories
    train_files = os.listdir(original_class_train_dir)
    test_files = os.listdir(original_class_test_dir)

    # Shuffling the order of the files
    random.shuffle(train_files)
    random.shuffle(test_files)

    # Moving the files to the reshuffled directories
    for file_name in train_files:
        src_path = os.path.join(original_class_train_dir, file_name)
        dst_path = os.path.join(reshuffled_class_train_dir, file_name)
        shutil.move(src_path, dst_path)

    for file_name in test_files:
        src_path = os.path.join(original_class_test_dir, file_name)
        dst_path = os.path.join(reshuffled_class_test_dir, file_name)
        shutil.move(src_path, dst_path)

print("Dataset reshuffled successfully.")
