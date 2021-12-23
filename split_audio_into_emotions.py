import os
import random
import shutil

from audio_utils import read_as_melspectrogram, normalize

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

random.seed(0)

source_dir = "audio_data"

test_data_size = None

if test_data_size != None:
    target_dir = "audio_data_emotions_" + str(test_data_size)
else:
    target_dir = "audio_data_emotions"


train_split = 0.7
val_split = 0.15
test_split = 0.15

emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

data_types = ["train", "val", "test"]

emotion_paths = {key: [] for key in emotions}

files = 0

for path in sorted(os.listdir(source_dir)):
    if not "." in path:
        if test_data_size != None:
            for file in sorted(os.listdir(os.path.join(source_dir, path))[:int(test_data_size*len(os.listdir(os.path.join(source_dir, path))))]):
                if file.split("-")[0] == "03":
                    print(file.split("-"))
                    emotion_index = int(file.split("-")[2].strip("0"))
                    emotion_paths[emotions[emotion_index - 1]].append(os.path.join(source_dir, path, file))
                    files += 1
        else:
            for file in sorted(os.listdir(os.path.join(source_dir, path))):
                if file.split("-")[0] == "03":
                    print(file.split("-"))
                    emotion_index = int(file.split("-")[2].strip("0"))
                    emotion_paths[emotions[emotion_index - 1]].append(os.path.join(source_dir, path, file))
                    files += 1

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
else:
    os.mkdir(target_dir)

trainval_data = {k: [] for k in emotions}
test_data = {k: [] for k in emotions}


for k, v in emotion_paths.items():
    l = v
    random.shuffle(l)
    trainval, test = l[:int((train_split + val_split)*len(emotion_paths[k]))], l[int((train_split + val_split)*len(emotion_paths[k])):]
    trainval_data[k] = trainval
    test_data[k] = test

for emotion in emotions:
    if not os.path.exists(os.path.join(target_dir, emotion)):
        os.mkdir(os.path.join(target_dir, emotion))
    for data_type in data_types:
        if not os.path.exists(os.path.join(target_dir, emotion, data_type)):
            os.mkdir(os.path.join(target_dir, emotion, data_type))

train_data = {k: [] for k in emotions}
val_data = {k: [] for k in emotions}

for k, v in trainval_data.items():
    l = v
    random.shuffle(v)
    train, val = l[:int((train_split + val_split)*len(trainval_data[k]))], l[int((train_split + val_split)*len(trainval_data[k])):]
    train_data[k] = train
    val_data[k] = val


print("Moving over training data...")
for k, v in tqdm(train_data.items(), total=len(train_data.items())):
    print(k, len(v))
    for path in v:
        dst_path = os.path.join(target_dir, k, "train", path.split("/")[-1].split(".")[0] + ".png")
        if not os.path.exists(dst_path):
            tmp_img = read_as_melspectrogram(path)
            plt.imsave(dst_path, normalize(tmp_img))

print("Moving over testing data...")
for k, v in tqdm(test_data.items(), total=len(test_data.items())):
    print(k, len(v))
    for path in v:
        dst_path = os.path.join(target_dir, k, "test", path.split("/")[-1].split(".")[0] + ".png")
        if not os.path.exists(dst_path):
            tmp_img = read_as_melspectrogram(path)
            plt.imsave(dst_path, normalize(tmp_img))

print("Moving over validation data...")
for k, v in tqdm(val_data.items(), total=len(val_data.items())):
    print(k, len(v))
    for path in v:
        dst_path = os.path.join(target_dir, k, "val", path.split("/")[-1].split(".")[0] + ".png")
        if not os.path.exists(dst_path):
            tmp_img = read_as_melspectrogram(path)
            plt.imsave(dst_path, normalize(tmp_img))

