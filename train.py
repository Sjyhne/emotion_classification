import tensorflow as tf

from model import build_model

from tensorflow.keras.applications.efficientnet import EfficientNetB0
import keras

from data_generator import EmotionDataset

# Utility for running experiments.
def run_experiment(train, test, val, model, epochs):
    
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    print(train[0][0].shape, train[0][1].shape)

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
    
    train_data = EmotionDataset("data_emotions_0.02", "train")
    test_data = EmotionDataset("data_emotions_0.05", "test")
    validation_data = EmotionDataset("data_emotions_0.05", "val")

    model = build_model(num_classes=8)

    h, sequence_model = run_experiment(train_data, test_data, validation_data, model, epochs=2)