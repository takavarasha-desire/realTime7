from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from cnn_model import second_model

input_shape = (256, 256, 1)  # Matching the input shape of the pre-trained model
num_classes = 7

# Loading the pre-trained model weights
pretrained_model = second_model(input_shape, num_classes)

pretrained_model.load_weights('ferg256_initial_model.keras')

# Preparing the Data
train_dir = "./fer2013_data/train"
validation_dir = "./fer2013_data/test"

# Data augmentation for the new dataset
new_train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

new_validation_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# Load and preprocess new dataset
batch_size = 10
target_size = (256, 256)  # Resizing to match the input shape of pre-trained model

new_train_generator = new_train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical'
)

new_validation_generator = new_validation_datagen.flow_from_directory(
    validation_dir,
    color_mode='grayscale',
    batch_size=batch_size,
    target_size=target_size,
    class_mode='categorical'
)

# Compiling the model
pretrained_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

# Fine-tuning the model with new dataset
epochs = 20

pretrained_model.fit(new_train_generator,
                     steps_per_epoch=len(new_train_generator),
                     epochs=epochs,
                     validation_data=new_validation_generator,
                     validation_steps=len(new_validation_generator),
                     )


# Evaluating the fine-tuned model
loss, accuracy = pretrained_model.evaluate(new_validation_generator)
print(f"Fine-tuned validation accuracy: {accuracy*100:.2f}%")

# Saving the fine-tuned model
pretrained_model.save('pretrained_finetuned_model3.keras')
