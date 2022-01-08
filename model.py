from typing import List
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

# from typing import Tuple
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow import reduce_mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy

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


class SpeechModel:
    def __init__(self) -> None:
        # def __init__(self, input_shape: Tuple) -> None:
        # self.input_shape = input_shape
        print("Downloading ResNet Weights")
        # input_shape should be more than 32 in h and w : (64, 64, 3)
        self.resnet_layer = ResNet50V2(
            include_top=False, weights="imagenet", input_shape=(64, 64, 3)
        )
        self.lossFn = CategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(1e-4)
        self.conv_model = None  # Create model on call
        self.td_model = None  # Create model on call

    def create_conv_model(self):
        conv_input_layer = L.Input((64, 64, 3))
        resnet_output = self.resnet_layer(conv_input_layer)  # (2,2,2048)
        average_pool = L.AveragePooling2D((2, 2))(resnet_output)  # (1,1,2048)
        flatten = L.Flatten()(average_pool)  # (2048)

        conv_model = Model(conv_input_layer, flatten)

        return conv_model

    def model_summary(self):
        if self.td_model is None:
            print("Create model first by calling SpeechModel.create_model()")
            return None
        return self.td_model.summary()

    def create_model(self, input_shape: List):
        # td_input_layer = L.Input(self.input_shape)
        td_input_layer = L.Input(input_shape)
        self.conv_model = self.create_conv_model()
        td_conv_layer = L.TimeDistributed(self.conv_model)(
            td_input_layer
        )  # output: (8, 2048)
        td_bilstm = L.Bidirectional(L.LSTM(128, return_sequences=True))(
            td_conv_layer
        )  # (8, 256)

        # Attention layer, returns matmul(distribution, value)
        # distribution is of shape [batch_size, Tq, Tv] while value is of shape [batch_size, Tv, dim]
        # The inner dimentinons except batch_size are same, we get output of dimention [batch_size, tq, dim]
        # Here, our Query and Value dimentions are 8, 256. That is, Tv, Tq = 8 and dim = 256
        # Final output of attention layer is [batch_size, 8, 256]
        bilstm_attention_seq = L.Attention(use_scale=True)(
            [td_bilstm, td_bilstm]
        )  # (8, 256)
        bilstm_attention = reduce_mean(
            bilstm_attention_seq, axis=-2
        )  # Calculate mean along each sequence
        # There is some error in this attention layer (could be the reason loss is going to nan)

        # These dimentions are changed due to the different conv model being used.
        td_dense = L.Dense(256, activation="relu")(bilstm_attention)
        td_dense = L.Dropout(0.25)(td_dense)
        td_dense = L.Dense(128, activation="relu")(td_dense)
        td_dense = L.Dense(128, activation="relu")(td_dense)
        td_dense = L.Dropout(0.25)(td_dense)

        td_output_layer = L.Dense(8)(td_dense)

        td_model = Model(td_input_layer, td_output_layer)

        # Compile here
        td_model.compile(optimizer=self.optimizer, loss=self.lossFn, metrics=["acc", "categorical_accuracy"])
        self.td_model = td_model
        return self.td_model


if __name__ == "__main__":
    SP = SpeechModel()
    model = SP.create_model()
    print(model.summary())