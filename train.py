import tensorflow as tf

from model import build_efficientnet, build_feature_model, build_inception, build_lstm, build_vgg, build_resnet, build_bilstm, SpeechModel

from tensorflow import keras

import json

from data_generator import AudioFeatureDataset, EmotionDataset, AudioEmotionDatasetV2

# Utility for running experiments.
def run_experiment(train, test, val, model, epochs, batchsize):
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=10, min_lr=0.00005)

    mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)

    seq_model = model
    history = seq_model.fit(
        train,
        epochs=epochs,
        validation_data=val,
        callbacks=[reduce_lr, mc],
    )

    #seq_model.load_weights(filepath)
    _, cat_acc = seq_model.evaluate(test)
    
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test categorical_accuracy: {round(cat_acc * 100, 2)}%")

    return history, seq_model


if __name__ == "__main__":
    
    dataset_name = "vggish_features_0.2"
    
    bsize = 128
    
    train_data = AudioFeatureDataset(dataset_name, bsize, "train")
    test_data = AudioFeatureDataset(dataset_name, bsize, "test")
    validation_data = AudioFeatureDataset(dataset_name, bsize, "val")

    #inp_shape = train_data[0][0].shape[1:]

    #print("inp_shape:", inp_shape)
    
    input_shape = (train_data[0][0].shape)
    num_classes = 6
    print("input_shape:", input_shape)

    epochs = 75

    model = build_feature_model(num_classes, input_shape)
    
    #print(train_data[0][0].shape)

    h, sequence_model = run_experiment(train_data, test_data, validation_data, model, epochs=epochs, batchsize=bsize)
    
    print(h.history)
    print(h)
    
    json.dump(str(h.history), open(f"results/{dataset_name}/blstm_history_rms_{epochs}.json", 'w'))
    
    