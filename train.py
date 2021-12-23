import tensorflow as tf

from model import build_efficientnet, build_inception, build_vgg, build_resnet

from tensorflow import keras

import json

from data_generator import EmotionDataset, AudioEmotionDataset

# Utility for running experiments.
def run_experiment(train, test, val, model, epochs):
    
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = model
    history = seq_model.fit(
        train,
        epochs=epochs,
        validation_data=val,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate(test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


if __name__ == "__main__":
    
    dataset_name = "audio_data_emotions_0.1"
    
    train_data = AudioEmotionDataset(dataset_name, "train")
    test_data = AudioEmotionDataset(dataset_name, "test")
    validation_data = AudioEmotionDataset(dataset_name, "val")

    img_size = (train_data.img_h, train_data.img_w)
    
    epochs = 10

    model = build_efficientnet(num_classes=8, img_size=img_size)

    h, sequence_model = run_experiment(train_data, test_data, validation_data, model, epochs=epochs)
    
    json.dump(h.history, open(f"results/audio/{dataset_name}/efficientnet_history_{epochs}_imagenet.json", 'w'))
    
    