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

import tensorflow as tf

random.seed(0)

source_dir = "audio_data"

SEQ_LEN = 4

test_data_size = 0.2

images = True

v2 = True

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


train_split = 0.8
val_split = 0.10
test_split = 0.10

emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

emotion_to_index = {val: id for id, val in enumerate(emotions)}
index_to_emotion = {id: val for id, val in enumerate(emotions)}

data_types = ["train", "val", "test"]

emotion_paths = {key: [] for key in emotions}

files = 0

for path in sorted(os.listdir(source_dir)):
    print(path)

    if not "." in path and not "OAF" in path:
        if test_data_size != None:
            for file in sorted(os.listdir(os.path.join(source_dir, path))[:int(test_data_size*len(os.listdir(os.path.join(source_dir, path))))]):
                if file.split("-")[0] == "03" and file.split("-")[5] == "01":
                    emotion_index = int(file.split("-")[2].strip("0"))
                    emotion_paths[emotions[emotion_index - 1]].append(os.path.join(source_dir, path, file))
        else:
            for file in sorted(os.listdir(os.path.join(source_dir, path))):
                if file.split("-")[0] == "03" and file.split("-")[5] == "01":
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
srs = []
durations = []

for _, paths in tqdm(enumerate(emotion_paths.values()), total=len(emotion_paths.values())):
    for path in paths:
        try:
            x = AudioSegment.from_file(path)
            twav, sr = librosa.load(path=path, sr=None)
            # Transform the normalized audio to np.array of samples.
            audio_clip_values.append(len(x.get_array_of_samples()))
            srs.append(sr)
            durations.append(librosa.get_duration(twav, sr))
        except:
            ...

longest_audio_clip = max(audio_clip_values)
duration_audio_clip = max(durations)

print("srs:", list(set(srs)))

print("longest audio clip:", longest_audio_clip)
print("longest duration for audio clip:", duration_audio_clip)
print(srs[audio_clip_values.index(longest_audio_clip)])

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

train_paths = []
for k, v in train_data.items():
    for path in v:
        train_paths.append(path)

test_paths = []
for k, v in test_data.items():
    for path in v:
        test_paths.append(path)

val_paths = []
for k, v in val_data.items():
    for path in v:
        val_paths.append(path)

assert len(train_paths) == len(list(set(train_paths)))
assert len(test_paths) == len(list(set(test_paths)))
assert len(val_paths) == len(list(set(val_paths)))
print("No multiple datapoints in same datatype")

for path in train_paths:
    assert path not in test_paths
    assert path not in val_paths

for path in test_paths:
    assert path not in train_paths
    assert path not in val_paths

for path in val_paths:
    assert path not in train_paths
    assert path not in test_paths

print("No dataleakage between datatypes")


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

def load_wav(file_path):
    wav, sr = librosa.load(file_path, mono=True)

    pre_emp = 0.97

    wav = np.append(wav[0], wav[1:] - pre_emp * wav[:-1])

    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    sr = tf.convert_to_tensor(sr, dtype=tf.float32)

    return wav, sr

def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size // size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(next(iterator))
        if remain:
            result[i].append(next(iterator))
            remain -= 1
    return result

def create_spectrogram_and_save_images_v2(data, datatype):
    print("Creating image dataset V2")
    print(f"Moving over {datatype} data..")

    for k, v in tqdm(data.items(), total=len(data.items())):
        for path in v:
            dst_path = os.path.join(target_dir, k, datatype, path.split("/")[-1].split(".")[0] + ".npy")
            if not os.path.exists(dst_path):
                wav, sr = load_wav(path)
                framed_log_mels = get_framed_mel_spectrograms(wav, sr)
                np_log_mels = np.asarray(framed_log_mels)
                ceil = int(np.ceil(np_log_mels.shape[0] / SEQ_LEN))
                floor = int(np.floor(np_log_mels.shape[0] / ceil))
                for idx in range(ceil):
                    list_log_mels = np_log_mels.tolist()
                    res = slice_list(list_log_mels, ceil)
                    for tmp in res:
                        tmptmp = tmp
                        for _ in range(SEQ_LEN - len(tmp)):
                            tmptmp.append(np.zeros(np.asarray(tmp).shape[1:]))
                    
                        assert len(tmp) == 4
                        print(len(tmp))

                        tmp = np.asarray(tmp)
                        p = dst_path.split(".")
                        p = ".".join(p[:-1]) + f"_{idx}." + p[-1]
                        np.save(dst_path, tmp)

def get_framed_mel_spectrograms(wav, sr=22050):
    # The duration of clips is 3 seconds, ie. 3000 miliseconds. Do some quick math to figure out frame_length.
    frame_length = tf.cast(sr * (25 / 1000), tf.int32)  # 25 ms
    frame_step = tf.cast(sr * (10 / 1000), tf.int32)  # 10 ms
    stft_out = tf.signal.stft(
        wav,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hamming_window,
    )
    num_spectrogram_bins = tf.shape(stft_out)[-1]
    stft_abs = tf.abs(stft_out)
    lower_edge_hz, upper_edge_hz = 20.0, 8000.0
    num_mel_bins = 64
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_edge_hz, upper_edge_hz
    )
    mel_spectrograms = tf.tensordot(stft_abs, linear_to_mel_weight_matrix, 1)

    # mel_spectrograms.set_shape(
    #     stft_abs.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    # )

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    log_mel_d1 = log_mel_spectrograms - tf.roll(log_mel_spectrograms, -1, axis=0)
    log_mel_d2 = log_mel_d1 - tf.roll(log_mel_d1, -1, axis=0)

    log_mel_three_channel = tf.stack(
        [log_mel_spectrograms, log_mel_d1, log_mel_d2], axis=-1
    )

    framed_log_mels = tf.signal.frame(
        log_mel_three_channel, frame_length=64, frame_step=32, pad_end=False, axis=0
    )

    return framed_log_mels

def construct_and_save_features(data, datatype):

    # Initialize variables
    total_length = longest_audio_clip # desired frame length for all of the audio samples.
    preferred_cut_length = 88200 # = 1 second | Most common sampling rates are 44.1 kHz and 48 kHz
    target_sr = 44100
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
            # Fetch the sample rate.
            if "OAF" not in path and path.split("/")[-1].split("-")[5] == "02":
                print("PATH:", path)
            _, sr = librosa.load(path=path, sr = None)
            # Load the audio file.
            try:
                rawsound = AudioSegment.from_file(path)
            except:
                print("Corrupted file:", path)
                corruped_paths.append(path)
            # Normalize the audio to +5.0 dBFS.
            array_sound = np.array(rawsound.get_array_of_samples())
            norm = np.linalg.norm(array_sound)
            normalizedsound = array_sound/norm
            # Transform the normalized audio to np.array of samples.
            normal_x = np.array(normalizedsound, dtype='float32')
            resampled_sound = librosa.resample(normal_x, sr, target_sr)
            # Trim silence from the beginning and the end.
            xt, index = librosa.effects.trim(resampled_sound, top_db=20)
            # Pad for duration equalization.
            divisable = int(np.ceil(len(xt) / preferred_cut_length))

            for i in range(divisable):
                rms = []
                zcr = []
                mfcc = []
                #poly = []
                #centroid = []
                #rolloff = []
                emotions = []
                segment = xt[i*preferred_cut_length:(i+1)*preferred_cut_length]
                if len(segment) != preferred_cut_length:
                    segment = np.pad(segment, (0, preferred_cut_length-len(segment)), 'constant')
                segment = nr.reduce_noise(y=segment, sr=target_sr)
                
                # Features extraction 
                f1 = librosa.feature.rms(segment, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square   
                f2 = librosa.feature.zero_crossing_rate(segment, frame_length=frame_length, hop_length=hop_length, center=True) # ZCR      
                f3 = librosa.feature.mfcc(segment, sr=target_sr, n_mfcc=15, hop_length=hop_length) # MFCC
                #f4 = librosa.feature.poly_features(segment, n_fft=frame_length, sr=target_sr, hop_length=hop_length, order=2)
                #f5 = librosa.feature.spectral_centroid(segment, n_fft=frame_length, sr=target_sr, hop_length=hop_length)
                #f6 = librosa.feature.spectral_rolloff(segment, n_fft=frame_length, sr=target_sr, hop_length=hop_length)

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
                #poly.append(f4)
                #centroid.append(f5)
                #rolloff.append(f6)
                emotions.append(label)



                f_rms = np.asarray(rms).astype('float32')
                f_rms = np.swapaxes(f_rms,1,2)
                f_zcr = np.asarray(zcr).astype('float32')
                f_zcr = np.swapaxes(f_zcr,1,2)
                f_mfccs = np.asarray(mfcc).astype('float32')
                f_mfccs = np.swapaxes(f_mfccs,1,2)
                #f_polys = np.asarray(poly).astype('float32')
                #f_polys = np.swapaxes(f_polys, 1, 2)
                #f_centroid = np.asarray(centroid).astype('float32')
                #f_centroid = np.swapaxes(f_centroid, 1, 2)
                #f_rolloff = np.asarray(rolloff).astype('float32')
                #f_rolloff = np.swapaxes(f_rolloff, 1, 2)

                #print('ZCR shape:',f_zcr.shape)
                #print('RMS shape:',f_rms.shape)
                #print('MFCCs shape:',f_mfccs.shape)
                #print('Polys shape:',f_polys.shape)
                #print('Centroid shape:',f_centroid.shape)
                #print('Rolloff shape:', f_rolloff.shape)

                # Concatenating all features to 'X' variable.
                X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

                # Preparing 'Y' as a 2D shaped variable.
                Y = np.asarray(emotions).astype('int8')
                Y = np.expand_dims(Y, axis=1)

                # Save X,Y arrays as lists to json files.

                x_data = X.tolist() 
                x_path = f'{target_dir}/{datatype}/features/{path.split(".")[0].split("/")[-1]}_{i}.json' # FILE SAVE PATH
                dump(x_data, open(x_path, "w"))

                y_data = Y.tolist() 
                y_path = f'{target_dir}/{datatype}/labels/{path.split(".")[0].split("/")[-1]}_{i}.json' # FILE SAVE PATH
                dump(y_data, open(y_path, "w"))

if images and v2:
    print("Creating spectrograms")

    create_spectrogram_and_save_images_v2(train_data, "train")
    create_spectrogram_and_save_images_v2(test_data, "test")
    create_spectrogram_and_save_images_v2(val_data, "val")

elif images and not v2:
    print("Creating images")
    
    create_spectrogram_and_save_images(train_data, "train")
    create_spectrogram_and_save_images(test_data, "test")
    create_spectrogram_and_save_images(val_data, "val")

else:
    print("Creating audio features")

    construct_and_save_features(train_data, "train")
    construct_and_save_features(test_data, "test")
    construct_and_save_features(val_data, "val")