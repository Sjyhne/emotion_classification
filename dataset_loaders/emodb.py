import os
import random

from .utils import get_actor_split, get_duration_span

import librosa
import numpy as np

# Removing boredom and disgust

EMOTION_MAP = {
    "W": "anger",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

DATA_DIR = "datasets/EMO-DB/wav"

TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2

def get_file_paths(datadir):
    filepaths = []
    for filepath in os.listdir(DATA_DIR):
        filepaths.append(os.path.join(DATA_DIR, filepath))
    
    return filepaths

def get_actors(filepaths):
    actors = []
    for filepath in filepaths:
        actors.append(filepath.split("/")[-1][:2])
    
    return list(set(actors))

def get_datatype_paths(filepaths):
    train_paths, val_paths, test_paths = [], [], []

    random.shuffle(filepaths)

    train_actors, val_actors, test_actors = get_actor_split(get_actors(filepaths), [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT])

    for path in filepaths:
        if not get_emodb_label(path) == "N/A":
            actor = path.split("/")[-1][:2]
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
        label = get_emodb_label(filepath)
        if label != "N/A":
            class_diversity[label] += 1
    
    return class_diversity


def get_data(datadir):
    
    filepaths = get_file_paths(datadir)

    class_diversity = get_class_diversity(filepaths)

    train_paths, val_paths, test_paths = get_datatype_paths(filepaths)

    print("EMO-DB class diversity:", class_diversity)

    print("EMO-DB file duration span:", get_duration_span(filepaths))

    print("Datatype sizes (train, val, test):", len(train_paths), "|", len(val_paths), "|", len(test_paths))
    print()
    return train_paths, val_paths, test_paths
    
def get_emodb_data():
    return get_data(DATA_DIR)

def get_emodb_label(filepath):
    filename = filepath.split("/")[-1]
    emotion_letter = filename.split(".")[0][5]
    try:
        emotion = EMOTION_MAP[emotion_letter]
    except:
        emotion = "N/A"

    return emotion