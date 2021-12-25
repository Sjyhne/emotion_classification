import tensorflow as tf

from model import build_efficientnet, build_inception, build_lstm, build_vgg, build_resnet

from tensorflow import keras

import json

from data_generator import AudioFeatureEmotionDataset, EmotionDataset, AudioEmotionDataset

# Utility for running experiments.
def run_experiment(train, test, val, model, epochs, batchsize):
    
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = model
    history = seq_model.fit(
        train,
        epochs=epochs,
        batch_size=batchsize,
        validation_data=val,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate(test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


if __name__ == "__main__":
    
    dataset_name = "audio_data_emotions_features_0.1"
    
    bsize = 4
    
    train_data = AudioFeatureEmotionDataset(dataset_name, bsize, "train")
    test_data = AudioFeatureEmotionDataset(dataset_name, bsize, "test")
    validation_data = AudioFeatureEmotionDataset(dataset_name, bsize, "val")

    inp_shape = train_data[0][0].shape[1:]

    print("inp_shape:", inp_shape)

    epochs = 100

    model = build_lstm(num_classes=8, inp_shape=inp_shape)
    
    print(train_data[0][0].shape)

    h, sequence_model = run_experiment(train_data, test_data, validation_data, model, epochs=epochs, batchsize=bsize)
    
    json.dump(h.history, open(f"results/audio/{dataset_name}/efficientnet_history_{epochs}_imagenet.json", 'w'))
    
    