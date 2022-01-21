import os
import random

import dataset_loaders.utils as utils

import librosa
import numpy as np

# Using all emotions

EMOTION_MAP = {
    "ANG": "anger",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

DATA_DIR = "datasets/CREMA-D/AudioWAV"

TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2

def get_file_paths(datadir):
    filepaths = []
    for filepath in os.listdir(datadir):
        filepaths.append(os.path.join(datadir, filepath))

    return filepaths

def get_actors(filepaths):
    actors = []
    for filepath in filepaths:
        actors.append(filepath.split("/")[-1][:4])
    
    return list(set(actors))

def get_datatype_paths(filepaths):
    train_paths, val_paths, test_paths = [], [], []

    random.shuffle(filepaths)

    train_actors, val_actors, test_actors = utils.get_actor_split(get_actors(filepaths), [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT])

    for path in filepaths:
        if not get_cremad_label(path) == "N/A":
            actor = path.split("/")[-1][:4]
            if actor in train_actors:
                train_paths.append(path)
            elif actor in val_actors:
                val_paths.append(path)
            else:
                test_paths.append(path)
        
    return train_paths, val_paths, test_paths
    
def get_class_diversity(filepaths):
    class_diversity = {v: 0 for v in EMOTION_MAP.values()}
    for filepath in filepaths:
        label = get_cremad_label(filepath)
        if label != "N/A":
            class_diversity[label] += 1
    
    return class_diversity

def get_data(datadir):
    
    filepaths = get_file_paths(datadir)

    class_diversity = get_class_diversity(filepaths)

    train_paths, val_paths, test_paths = get_datatype_paths(filepaths)

    print("CREMA-D class diversity:", class_diversity)

    print("CREMA-D file duration span:", utils.get_duration_span(filepaths))

    print("Datatype sizes (train, val, test):", len(train_paths), "|", len(val_paths), "|", len(test_paths))
    print()
    return train_paths, val_paths, test_paths
    
def get_cremad_data():
    return get_data(DATA_DIR)

def get_cremad_label(filepath):
    filename = filepath.split("/")[-1]
    emotion_letter = filename.split(".")[0].split("_")[2]
    try:
        emotion = EMOTION_MAP[emotion_letter]
    except:
        emotion = "N/A"

    return emotion