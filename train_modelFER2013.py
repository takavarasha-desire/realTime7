import os
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator
from cnn_models import first_model, second_model, third_model, fifth_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Preparing the Data
train_dir = 'fer2013_data/train'
validation_dir = 'fer2013_data/test'

# Data augmentation to improve generalization
train_datagen = ImageDataGenerator(
    #rescale=1.0/255,
    rotation_range=2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    #horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator()

batch_size = 32
img_height, img_width = 48, 48

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    color_mode='grayscale',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

input_shape = (48, 48, 1)
num_classes = 7
dataset_dir = train_dir

X_train = []  # List to store image data
y_train = []  # List to store labels

# Iterating through subdirectories (each representing a class)
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for image_filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_filename)
            # Read and preprocess the image (you may need to resize and normalize)
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (48, 48))  # Resize to your desired dimensions
            image = image / 255.0  # Normalize pixel values
            X_train.append(image)
            y_train.append(class_name)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Computing class weights based on the training dataset
class_labels = np.unique(y_train)
class_label_map = {class_name: label for label, class_name in enumerate(class_labels)}
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# model = first_model(input_shape, num_classes)
# model = second_model(input_shape, num_classes)
# model = third_model(input_shape, num_classes)
model = fifth_model(input_shape, num_classes)
# model = model_with_leaky_relu(input_shape, num_classes)

# Compiling the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
epochs = 20

num_train = 28709
num_val = 7178

history = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size,
    class_weight=class_weight_dict
)

# Evaluating Model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
validation_generator.reset()  # Resetting generator to start from the beginning
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validation_generator.classes, y_pred)

# Plotting Confusion Matrix as a Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(validation_generator.classes, y_pred, target_names=class_labels)
print(report)

# Plotting Training & Validation Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')

# Plotting Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.show()

# Saving the model to disk
model.save('fer2013_first_model.keras')

model_json = model.to_json()

# Saving the JSON to a file
with open("fer2013_first_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("fer2013_first_model.keras")
