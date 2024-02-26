from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU
from keras.layers import BatchNormalization, Dense, Dropout, Activation, AveragePooling2D

num_features = 64

def initial_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.50))

    model.add(Flatten())

    model.add(Dense(2*2*2*num_features, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(2*2*num_features, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(2*num_features, activation='relu'))
    model.add(Dropout(0.50))

    model.add(Dense(num_classes, activation='softmax'))
    return model
#######################################################################################
def regularized_model(input_shape, num_classes, l2_weight=0.01, dropout_rate=0.10):
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=l2(l2_weight)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    #model.add(Dropout(dropout_rate))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l2_weight)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(dropout_rate))

    # Convolutional Layer 3
    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l2_weight)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(dropout_rate))
    
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(343, activation='relu', kernel_regularizer=l2(l2_weight)))
    
    # Fully Connected Layer 2
    model.add(Dense(49, activation='relu', kernel_regularizer=l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


#######################################################################################
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


#######################################################################################
def atrous_model(input_shape, num_classes):
    model = Sequential()
        
    # Atrous Convolution Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))

    # Atrous Convolution Layer 2
    model.add(Conv2D(64, (3, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))

    # Atrous Convolution Layer 3
    model.add(Conv2D(64, (3, 3), dilation_rate=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.10))
    model.add(AveragePooling2D((2, 2)))
    
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

########################################################################################
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

########################################################################################

def second_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))


    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def third_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def fourth_model(input_shape, num_classes):
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

def fifth_model(input_shape, num_classes):
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

def sixth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def seventh_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def eighth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def ninth_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(num_classes, activation='softmax'))
    return model
