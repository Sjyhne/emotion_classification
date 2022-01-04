from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import optimizers

from tensorflow.keras.optimizers import RMSprop, Adam

from data_generator import IMG_SIZE
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Layer, Bidirectional, LSTM, Attention, ELU, BatchNormalization
from tensorflow.keras import Model, Sequential
import tensorflow.keras as K
import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.constraints as constraints

def build_efficientnet(num_classes, img_size=(IMG_SIZE, IMG_SIZE), channels=3):
    
    base_model = EfficientNetB7(input_shape = (img_size[0], img_size[1], channels), include_top = False, weights='imagenet')
    
    for layer in base_model.layers:
        layer.trainable = True
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    predictions = Dense(num_classes, activation="softmax")(x)
    model_final = Model(inputs=base_model.input, outputs=predictions)
    
    model_final.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model_final

def build_vgg(num_classes):

    base_model = VGG16(input_shape = (IMG_SIZE, IMG_SIZE, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False
        
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.3)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])
    
    return model

def build_inception(num_classes):
    
    base_model = InceptionV3(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top = False, weights = 'imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
    
    return model

def build_resnet(num_classes, img_size=(IMG_SIZE, IMG_SIZE)):

    res_model = ResNet50(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights=None, pooling="max")
    
    for layer in res_model.layers:
        layer.trainable = True
    
    base_model = Sequential()
    base_model.add(res_model)
    base_model.add(Dense(num_classes, activation='softmax'))
    
    base_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
    
    return base_model

    
def build_lstm(num_classes, inp_shape):
    model = Sequential()
    model.add(Bidirectional(layers.LSTM(256, return_sequences=True), input_shape=(inp_shape)))
    model.add(layers.LSTM(256))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    print(model.summary())

    # Compile & train   
    model.compile(loss='categorical_crossentropy', 
                    optimizer=tf.keras.optimizers.RMSprop(),
                    metrics=['categorical_accuracy'])

    return model

def build_bilstm(num_classes, inp_shape):
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(inp_shape)))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(ELU())
    model.add(Dropout(0.2)) 
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="RMSProp",
                  metrics=['accuracy', "categorical_accuracy"])
    
    return model