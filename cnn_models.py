from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dense
from keras.layers import BatchNormalization, Dense, Dropout, Activation, AveragePooling2D

from keras.regularizers import l1, l2

def first_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.30))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.50))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def first_model1(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=l2(0.00), activity_regularizer=l1(0.00)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.00), activity_regularizer=l1(0.00)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.00), activity_regularizer=l1(0.00)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def first_model2(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Conv2D(512, (3, 3), kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.10))
    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.90), activity_regularizer=l1(0.99)))
    return model


def second_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(640, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def third_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def fourth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    return model



def fifth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def sixth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def seventh_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.50))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_with_batch_normalization(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())  
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())  
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())  
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(Dropout(0.1))
    
    model.add(Dense(num_classes, activation='softmax'))
    return model

def atrous_model(input_shape, num_classes):
    model = Sequential()
        
    # Atrous Convolution Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, dilation_rate=2))
    model.add(Conv2D(64, (3, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))

    # Atrous Convolution Layer 2
    model.add(Conv2D(96, (3, 3), dilation_rate=2))
    model.add(Conv2D(128, (3, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))

    # Atrous Convolution Layer 3
    model.add(Conv2D(192, (3, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))
    
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_with_leaky_relu(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))  
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))  
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))  
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))  
    model.add(Dropout(0.40))
    
    model.add(Dense(num_classes, activation='softmax'))
    return model

def vgg_based_model(input_shape, num_classes):
    # Create a Sequential model
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # 1000 is the default number of classes for ImageNet
    return model
