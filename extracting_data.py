import os
import zipfile

# Extracting data from the zip folder with password
with zipfile.ZipFile("C:\\Users\\takav\\PycharmProjects\\FERG_DB_256.zip", "r") as zip_ref:
    zip_ref.setpassword(b"UWFERGdb2016")
    zip_ref.extractall('extracted_data')


data_dir = 'extracted_data'

outer_names = ['test', 'train']
inner_class_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise.']
os.makedirs(data_dir, exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('extracted_data', outer_name), exist_ok=True)
    for inner_class_name in inner_class_names:
        os.makedirs(os.path.join('extracted_data', outer_name, inner_class_name), exist_ok=True)
