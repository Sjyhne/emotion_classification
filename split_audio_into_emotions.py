import os
import random
import shutil

from audio_utils import read_as_melspectrogram, normalize

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from json import dump

from pydub import AudioSegment, effects
import noisereduce as nr
import librosa
random.seed(0)

source_dir = "audio_data"

test_data_size = 0.1

images = False

if images == True:
    if test_data_size != None:
        target_dir = "audio_data_emotions_images_" + str(test_data_size)
    else:
        target_dir = "audio_data_emotions_images"
else:
    if test_data_size != None:
        target_dir = "audio_data_emotions_features_" + str(test_data_size)
    else:
        target_dir = "audio_data_emotions_features"


train_split = 0.7
val_split = 0.15
test_split = 0.15

emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

emotion_to_index = {val: id for id, val in enumerate(emotions)}
index_to_emotion = {id: val for id, val in enumerate(emotions)}

tess_emotion_map = {"neutral", "01"}

data_types = ["train", "val", "test"]

emotion_paths = {key: [] for key in emotions}

files = 0

for path in sorted(os.listdir(source_dir)):
    print(path)

    if not "." in path and not "OAF" in path:
        if test_data_size != None:
            for file in sorted(os.listdir(os.path.join(source_dir, path))[:int(test_data_size*len(os.listdir(os.path.join(source_dir, path))))]):
                if file.split("-")[0] == "03":
                    emotion_index = int(file.split("-")[2].strip("0"))
                    emotion_paths[emotions[emotion_index - 1]].append(os.path.join(source_dir, path, file))
        else:
            for file in sorted(os.listdir(os.path.join(source_dir, path))):
                if file.split("-")[0] == "03":
                    emotion_index = int(file.split("-")[2].strip("0"))
                    emotion_paths[emotions[emotion_index - 1]].append(os.path.join(source_dir, path, file))
    
    elif "OAF" in path and not "." in path:
        print("IN OAF")
        if test_data_size != None:
            for file in sorted(os.listdir(os.path.join(source_dir, path))[:int(test_data_size*len(os.listdir(os.path.join(source_dir, path))))]):
                if file.split("_")[-1].split(".")[0] == "ps":
                    emotion_paths["surprised"].append(os.path.join(source_dir, path, file))
                elif file.split("_")[-1].split(".")[0] == "fear":
                    emotion_paths["fearful"].append(os.path.join(source_dir, path, file))
                else:
                    print(file.split("_")[-1].split(".")[0])
                    emotion_paths[file.split("_")[-1].split(".")[0]].append(os.path.join(source_dir, path, file))
        else:
            print("IN NO TEST DATA SIZE")
            for file in sorted(os.listdir(os.path.join(source_dir, path))):
                print(file)
                if file.split("_")[-1].split(".")[0] == "ps":
                    emotion_paths["surprised"].append(os.path.join(source_dir, path, file))
                elif file.split("_")[-1].split(".")[0] == "fear":
                    emotion_paths["fearful"].append(os.path.join(source_dir, path, file))
                else:
                    print(file.split("_")[-1].split(".")[0])
                    emotion_paths[file.split("_")[-1].split(".")[0]].append(os.path.join(source_dir, path, file))


for key, value in emotion_paths.items():
    print(key, " - number of paths:", len(value))

audio_clip_values = []


for _, paths in tqdm(enumerate(emotion_paths.values()), total=len(emotion_paths.values())):
    for path in paths:
        try:
            x = AudioSegment.from_file(path)
            # Normalize the audio to +5.0 dBFS.
            normalizedsound = effects.normalize(x, headroom = 5.0)
            # Transform the normalized audio to np.array of samples.
            x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
            xt, index = librosa.effects.trim(x, top_db=30)
            audio_clip_values.append(len(xt))
        except:
            ...

longest_audio_clip = max(audio_clip_values)

print("longest audio clip:", longest_audio_clip)

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

def create_spectrogram_and_save_images(data, datatype):
    print("Creating image dataset")
    print(f"Moving over {datatype} data...")
    for k, v in tqdm(data.items(), total=len(data.items())):
        print(k, len(v))
        for path in v:
            dst_path = os.path.join(target_dir, k, datatype, path.split("/")[-1].split(".")[0] + ".png")
            if not os.path.exists(dst_path):
                tmp_img = read_as_melspectrogram(path)
                plt.imsave(dst_path, normalize(tmp_img))

def construct_and_save_features(data, datatype):

    # Initialize variables
    total_length = longest_audio_clip # desired frame length for all of the audio samples.
    frame_length = 2048
    hop_length = 512

    corruped_paths = []

    os.makedirs(os.path.join(target_dir, datatype, "features"))
    os.makedirs(os.path.join(target_dir, datatype, "labels"))


    print(f"Creating features for {datatype} data")
    for k, v in tqdm(data.items(), total=len(data.items())):
        print(k, len(v))
        # Initialize data lists
        for path in v:
            rms = []
            zcr = []
            mfcc = []
            emotions = []
            # Fetch the sample rate.
            _, sr = librosa.load(path=path, sr = None)
            # Load the audio file.
            try:
                rawsound = AudioSegment.from_file(path)
            except:
                print("Corrupted file:", path)
                corruped_paths.append(path)
            # Normalize the audio to +5.0 dBFS.
            normalizedsound = effects.normalize(rawsound, headroom = 5.0)
            # Transform the normalized audio to np.array of samples.
            normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
            # Trim silence from the beginning and the end.
            xt, index = librosa.effects.trim(normal_x, top_db=30)
            # Pad for duration equalization.
            print("total_length:", total_length, "- len(xt):", len(xt), "| path:", path)
            padded_x = np.pad(xt, (0, total_length-len(xt)), 'constant')
            # Noise reduction.
            final_x = nr.reduce_noise(y=padded_x, sr=sr)

            # Features extraction 
            f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square   
            f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length, center=True) # ZCR      
            f3 = librosa.feature.mfcc(final_x, sr=sr, n_mfcc=13, hop_length= hop_length) # MFCC

            if "OAF" in path:
                l = path.split("_")[-1].split(".")[0]
                if l == "ps":
                    label = emotion_to_index["surprised"]
                elif l == "fear":
                    label = emotion_to_index["fearful"]
                else:
                    label = emotion_to_index[l]
            else:
                label = int(path.split("-")[2].strip("0")) - 1

            # Filling the data lists  
            rms.append(f1)
            zcr.append(f2)
            mfcc.append(f3)
            emotions.append(label)



            f_rms = np.asarray(rms).astype('float32')
            f_rms = np.swapaxes(f_rms,1,2)
            f_zcr = np.asarray(zcr).astype('float32')
            f_zcr = np.swapaxes(f_zcr,1,2)
            f_mfccs = np.asarray(mfcc).astype('float32')
            f_mfccs = np.swapaxes(f_mfccs,1,2)

            print('ZCR shape:',f_zcr.shape)
            print('RMS shape:',f_rms.shape)
            print('MFCCs shape:',f_mfccs.shape)

            # Concatenating all features to 'X' variable.
            X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

            # Preparing 'Y' as a 2D shaped variable.
            Y = np.asarray(emotions).astype('int8')
            Y = np.expand_dims(Y, axis=1)

            # Save X,Y arrays as lists to json files.

            x_data = X.tolist() 
            x_path = f'{target_dir}/{datatype}/features/{path.split(".")[0].split("/")[-1]}.json' # FILE SAVE PATH
            dump(x_data, open(x_path, "w"))

            y_data = Y.tolist() 
            y_path = f'{target_dir}/{datatype}/labels/{path.split(".")[0].split("/")[-1]}.json' # FILE SAVE PATH
            dump(y_data, open(y_path, "w"))

if images:
    print("Creating images")
    
    create_spectrogram_and_save_images(train_data, "train")
    create_spectrogram_and_save_images(test_data, "test")
    create_spectrogram_and_save_images(val_data, "val")

else:
    print("Creating audio features")

    construct_and_save_features(train_data, "train")
    construct_and_save_features(test_data, "test")
    construct_and_save_features(val_data, "val")