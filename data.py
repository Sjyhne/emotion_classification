from cv2 import merge
import numpy as np
import matplotlib.pyplot as plt
import librosa
from dataset_loaders.emodb import get_emodb_data, get_emodb_label
from dataset_loaders.shemo import get_shemo_data, get_shemo_label
from dataset_loaders.emovo import get_emovo_data, get_emovo_label

from audio_preprocessing import create_spectrogram_and_save_images, load_wav, get_framed_mel_spectrograms

import random
import shutil
import os

EMOTION_MAP = {
    "anger": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}

def combine_datatypes(data):
    result_data = []
    for d in data:
        result_data.extend(d)

    return result_data

def get_percentage_of_data(data, data_percentage):
    new_data = []
    for d in data:
        if len(new_data) < int(len(data) * data_percentage):
            new_data.append(d)
        else:
            break
    return new_data

def merge_datasets(data_percentage=1.0):

    train_emovo, val_emovo, test_emovo = get_emovo_data()
    train_emodb, val_emodb, test_emodb = get_emodb_data()
    train_shemo, val_shemo, test_shemo = get_shemo_data()

    train_paths = combine_datatypes([train_emovo, train_emodb, train_shemo])
    val_paths = combine_datatypes([val_emovo, val_emodb, val_shemo])
    test_paths = combine_datatypes([test_emovo, test_emodb, test_shemo])

    print("Nr of audio-files (train, val, test):", len(train_paths), len(val_paths), len(test_paths))

    if data_percentage != 1.0:
        random.shuffle(train_paths)
        random.shuffle(val_paths)
        random.shuffle(test_paths)

        train_paths = get_percentage_of_data(train_paths, data_percentage)
        val_paths = get_percentage_of_data(val_paths, data_percentage)
        test_paths = get_percentage_of_data(test_paths, data_percentage)
    
        print("Nr of audio-files (train, val, test) after split:", len(train_paths), len(val_paths), len(test_paths))
    
    return train_paths, val_paths, test_paths

def create_spectogram_dataset(target_dir, split_val=1.0):

    if split_val != 1.0:
        target_dir = target_dir + "_" + str(split_val)

    datatypes = ["train", "val", "test"]

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    
    for datatype in datatypes:
        os.mkdir(os.path.join(target_dir, datatype))
    
    tr, va, te = merge_datasets(split_val)

    create_spectrogram_and_save_images(tr, "train", target_dir)
    create_spectrogram_and_save_images(va, "val", target_dir)
    create_spectrogram_and_save_images(te, "test", target_dir)
    

def get_label(path):
    if "EMOVO" in path:
        label = get_emovo_label(path)
    elif "ShEMO" in path:
        label = get_shemo_label(path)
    else:
        label = get_emodb_label(path)
    return EMOTION_MAP[label]


if __name__ == "__main__":
    create_spectogram_dataset("spectograms", 0.05)


