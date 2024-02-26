from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from cnn_model import initial_model

# Preparing the Data
train_dir = "./data_split/train"
validation_dir = "./data_split/test"

# Data augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

batch_size = 32
img_height, img_width = 256, 256
input_shape = (img_height, img_width, 1)

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

# Building the Model
num_classes = 7
model = initial_model(input_shape, num_classes)

# Compiling the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
epochs = 20

model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {accuracy*100:.2f}%")

# Saving the model to disk
model.save('ferg256_initial_model.keras')
model_json = model.to_json()

# Saving the JSON to a file
with open("ferg256_initial_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("ferg256_initial_model.keras")
