import os
from keras.preprocessing.image import ImageDataGenerator
from cnn_models import first_model, second_model, third_model, \
    fourth_model, fifth_model, seventh_model, atrous_model, vgg_based_model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Nadam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Preparing the Data
train_dir = 'fer2013_data/train'
validation_dir = 'fer2013_data/test'

# Data augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator()

batch_size = 256
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

# model = eighth_model(input_shape, num_classes)
# model = ninth_model(input_shape, num_classes)
# model = model_with_leaky_relu(input_shape, num_classes) # jacky convolution
# model = atrous_model(input_shape, num_classes)
# model = vgg_based_model(input_shape, num_classes)
model = first_model(input_shape, num_classes)
# model = second_model(input_shape, num_classes)
# model = third_model(input_shape, num_classes)   
# model = fourth_model(input_shape, num_classes)
# model = fifth_model(input_shape, num_classes)
# model = seventh_model(input_shape, num_classes)
# model = model_with_batch_normalization(input_shape, num_classes)

learning_rate = 0.0001

# Compiling the Model
model.compile(optimizer=Adam(learning_rate=learning_rate),
             loss='categorical_crossentropy',
            metrics=['accuracy'])

#optimizer=SGD(learning_rate=learning_rate)

#model.compile(optimizer=optimizer,
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])



# optimizer = RMSprop(learning_rate=learning_rate)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# optimizer = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# optimizer = Adagrad(learning_rate=learning_rate)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])




# Training the Model
epochs = 500         # was 113, can also be 141

num_train = 28709
num_val = 7178

history = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size,
    shuffle=True
)

# Evaluating Model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
validation_generator.reset()  # Reset generator to start from the beginning
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(validation_generator.classes, y_pred)

# Plot Confusion Matrix as a Heatmap
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

# # Saving the model to disk
# model.save('fer2013_1stModel.keras')

model_json = model.to_json()

# # Saving the JSON to a file
# with open("fer2013_1stModel.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("fer2013_1stModel.keras")


# Define the folder to save the models
models_folder = "./models"

# Create the folder if it doesn't exist
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# ...

# Saving the model to disk in the models folder
model.save(os.path.join(models_folder, 'fer2013_Model.keras'))

# Saving the JSON to a file in the models folder
json_path = os.path.join(models_folder, 'fer2013_Model.json')
with open(json_path, "w") as json_file:
    json_file.write(model_json)

# Saving the weights to the models folder
model.save_weights(os.path.join(models_folder, 'fer2013_Model_weights.keras'))