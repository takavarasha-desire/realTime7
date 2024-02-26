import os
import zipfile
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# from keras.preprocessing.image import ImageDataGenerator

# convert string to integer


def ascii_to_int(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n


with zipfile.ZipFile("C:\\Users\\takav\\PycharmProjects\\FER_2013.zip", "r") as zip_ref:
    zip_ref.extractall('fer2013_data')


data_dir = 'fer2013_data'

outer_names = ['test', 'train']
inner_class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise.']
os.makedirs(data_dir, exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('fer2013_data', outer_name), exist_ok=True)
    for inner_class_name in inner_class_names:
        os.makedirs(os.path.join('fer2013_data', outer_name, inner_class_name), exist_ok=True)

# You may need to adjust this based on your dataset structure

# to keep count of each category
angry = 0
disgust = 0
fear = 0
happy = 0
neutral = 0
sad = 0
surprise = 0
angry_test = 0
disgust_test = 0
neutral_test = 0
fear_test = 0
happy_test = 0
sad_test = 0
surprise_test = 0

df = pd.DataFrame(data_dir)
df.to_csv('./fer2013.csv', index=False)

df_read = pd.read_csv('./fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df_read['pixels'][i]
    words = txt.split()

    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = ascii_to_int(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('train/angry/im' + str(angry) + '.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save('train/disgusted/im' + str(disgust) + '.png')
            disgust += 1
        elif df['emotion'][i] == 2:
            img.save('train/fearful/im' + str(fear) + '.png')
            fear += 1
        elif df['emotion'][i] == 3:
            img.save('train/happy/im' + str(happy) + '.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save('train/sad/im' + str(sad) + '.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save('train/surprise/im' + str(surprise) + '.png')
            surprise += 1
        elif df['emotion'][i] == 6:
            img.save('train/neutral/im' + str(neutral) + '.png')
            neutral += 1

    # test
    else:
        if df['emotion'][i] == 0:
            img.save('test/angry/im' + str(angry_test) + '.png')
            angry_test += 1
        elif df['emotion'][i] == 1:
            img.save('test/disgust/im' + str(disgust_test) + '.png')
            disgust_test += 1
        elif df['emotion'][i] == 2:
            img.save('test/fear/im' + str(fear_test) + '.png')
            fear_test += 1
        elif df['emotion'][i] == 3:
            img.save('test/joy/im' + str(happy_test) + '.png')
            happy_test += 1
        elif df['emotion'][i] == 4:
            img.save('test/sadness/im' + str(sad_test) + '.png')
            sad_test += 1
        elif df['emotion'][i] == 5:
            img.save('test/surprise/im' + str(surprise_test) + '.png')
            surprise_test += 1
        elif df['emotion'][i] == 6:
            img.save('test/neutral/im' + str(neutral_test) + '.png')
            neutral_test += 1

print("Done!")

""""
train_dir = 'fer2013_data train'
validation_dir = 'fer2013_data test'


# Step 3: Use Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize pixel values between 0 and 1
validation_datagen = ImageDataGenerator(rescale=1.0/255)

batch_size = 32
img_height, img_width = 150, 150

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # 'categorical' for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 4: Preprocess and Train the Model
# ... (Define and compile your model here)
# model.fit(train_generator, validation_data=validation_generator, ...)
"""
