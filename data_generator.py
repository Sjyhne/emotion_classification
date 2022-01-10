import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from video_utils import IMG_SIZE, BATCH_SIZE

import tensorflow as tf

import math
import random

from tqdm import tqdm
import librosa

from json import load

from video_utils import load_video

from tensorflow.keras.utils import Sequence, to_categorical

from audio_utils import read_audio, read_as_melspectrogram, normalize

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

class EmotionDataset(Sequence):

    def __init__(self, path_to_videos, datatype="train") -> None:
        super().__init__()
        self.path_to_videos = path_to_videos
        self.datatype = datatype
        self.video_paths = self.get_all_filepaths()
        self.batchsize = BATCH_SIZE
        self.n_batches = self.calculate_batches()

        self.curr_file_index = 0
        self.curr_batch_index = 0
        self.curr_batch_array_size = 0

        self.batches = []
        self.labels = []

        print(self.n_batches)

    def get_all_filepaths(self):
        filepaths = []
        for label in os.listdir(self.path_to_videos):
            for file in os.listdir(os.path.join(self.path_to_videos, label, self.datatype)):
                filepaths.append(os.path.join(self.path_to_videos, label, self.datatype, file))
        
        random.shuffle(filepaths)
        return filepaths
    
    def calculate_batches(self):
        tot_batches = 0
        for _, file in tqdm(enumerate(self.video_paths), desc="Calculating n-batch", total=len(self.video_paths)):
            cap = cv2.VideoCapture(file)
            ret = True
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            
            n_batches = math.ceil(frame_count / self.batchsize)

            tot_batches += n_batches
        
        return tot_batches - 1
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        
        if self.curr_batch_index == self.curr_batch_array_size:
            self.curr_batch_index = 0
            self.batches, self.labels = load_video(self.video_paths[self.curr_file_index])
            self.curr_batch_array_size = self.batches.shape[0]
            self.curr_file_index += 1
        
        
        return_batches, return_labels = self.batches[self.curr_batch_index], self.labels[self.curr_batch_index]
        
        self.curr_batch_index += 1
        
        if self.curr_file_index == len(self.video_paths) - 1:
            self.curr_file_index = 0
            self.curr_batch_index = 0

        return return_batches, return_labels


class AudioEmotionDataset(Sequence):
    def __init__(self, path_to_images, batchsize, datatype="train") -> None:
        super().__init__()
        self.path_to_images = path_to_images
        self.datatype = datatype
        print("Creating AudioEmotionDataset from", self.path_to_images, "of datatype", self.datatype)
        print("Getting image paths")
        self.image_paths = self.get_image_paths()
        self.batchsize = batchsize
        print("Creating batches")
        self.image_batches = self.get_image_batches()

        self.img_h = 128
        self.img_w = 318
        self.channels = 3
    
        print("img_h:", self.img_h, "- img_w:", self.img_w, "- channels:", self.channels)


    
    def get_image_paths(self):
        image_paths = []
        for emotion in os.listdir(self.path_to_images):
            for file in os.listdir(os.path.join(self.path_to_images, emotion, self.datatype)):
                image_paths.append(os.path.join(self.path_to_images, emotion, self.datatype, file))
        
        random.shuffle(image_paths)
        return image_paths
    
    def get_image_batches(self):
        batches = []
        batch = []
        for path in self.image_paths:
            batch.append(path)
            if len(batch) == self.batchsize:
                batches.append(batch)
                batch = []
            
        if len(batch) != self.batchsize:
            while True:
                batch.append(random.choice(self.image_paths))
                if len(batch) >= self.batchsize:
                    print(len(batch), ">=", self.batchsize)
                    break
            
            batches.append(batch)
        
        return batches


    def __len__(self):
        return math.ceil(len(self.image_paths)/self.batchsize)

    def __getitem__(self, idx):
        image_batch = self.image_batches[idx]
        images = np.empty((self.batchsize, self.img_h, self.img_w, self.channels))
        labels = np.empty((self.batchsize, 8))
        for i, path in enumerate(image_batch):
            images[i] = plt.imread(path)[:, :, :3]
            l = int(path.split("/")[-1].split("-")[2].strip("0")) - 1
            tmp_label = [0 for i in range(8)]
            tmp_label[l] = 1
            labels[i] = np.array(tmp_label)

        
        return images, labels

class AudioEmotionDatasetV2(Sequence):
    def __init__(self, path_to_images, batchsize, datatype="train") -> None:
        super().__init__()
        self.path_to_images = path_to_images
        self.datatype = datatype
        print("Creating AudioEmotionDataset from", self.path_to_images, "of datatype", self.datatype)
        print("Getting image paths")
        self.image_paths = self.get_image_paths()
        self.batchsize = batchsize
        print("Creating batches")
        self.image_batches = self.get_image_batches()

        self.seq_len = 4
        self.img_h = 64
        self.img_w = 64
        self.channels = 3
    
        print("seq_len:", self.seq_len, "- img_h:", self.img_h, "- img_w:", self.img_w, "- channels:", self.channels)


    
    def get_image_paths(self):
        image_paths = []
        for emotion in os.listdir(self.path_to_images):
            for file in os.listdir(os.path.join(self.path_to_images, emotion, self.datatype)):
                image_paths.append(os.path.join(self.path_to_images, emotion, self.datatype, file))
        
        random.shuffle(image_paths)
        return image_paths
    
    def get_image_batches(self):
        batches = []
        batch = []
        for path in self.image_paths:
            batch.append(path)
            if len(batch) == self.batchsize:
                batches.append(batch)
                batch = []
            
        if len(batch) != self.batchsize:
            while True:
                batch.append(random.choice(self.image_paths))
                if len(batch) >= self.batchsize:
                    print(len(batch), ">=", self.batchsize)
                    break
            
            batches.append(batch)
        
        return batches


    def __len__(self):
        return math.ceil(len(self.image_paths)/self.batchsize)

    def get_label(self, path):
        emotion = path.split("/")[1]
        idx = EMOTIONS.index(emotion)
        return idx


    def __getitem__(self, idx):
        image_batch = self.image_batches[idx]
        images = np.empty((self.batchsize, self.seq_len, self.img_h, self.img_w, self.channels))
        labels = np.empty((self.batchsize, 8))
        for i, path in enumerate(image_batch):
            data = np.load(path)
            images[i] = data
            l = self.get_label(path)
            print("path:", path, "| label:", l)
            tmp_label = [0 for i in range(8)]
            tmp_label[l] = 1
            labels[i] = np.array(tmp_label)

        
        return images, labels

    
class AudioFeatureEmotionDataset(Sequence):
    def __init__(self, path_to_data: str, bsize: int, datatype: str) -> None:
        super().__init__()

        self.path_to_data = path_to_data
        self.datatype = datatype
        self.batch_size = bsize

        self.feature_paths = [os.path.join(self.path_to_data, self.datatype, "features", file) for file in sorted(os.listdir(os.path.join(self.path_to_data, self.datatype, "features")))]
        self.label_paths = [os.path.join(self.path_to_data, self.datatype, "labels", file) for file in sorted(os.listdir(os.path.join(self.path_to_data, self.datatype, "labels")))]
        self.shuffle_labels_and_data()
        self.create_batches()


    
    def shuffle_labels_and_data(self):
        zipped = list(zip(self.feature_paths, self.label_paths))
        random.shuffle(zipped)
        self.data_paths = []
        self.label_paths = []
        for item in zipped:
            self.data_paths.append(item[0])
            self.label_paths.append(item[1])
    
    def create_batches(self):
        self.feature_batches = []
        self.label_batches = []
        tmp_feature_batch = []
        tmp_label_batch = []
        for i in range(len(self.data_paths)):
            tmp_feature_batch.append(self.data_paths[i])
            tmp_label_batch.append(self.label_paths[i])
            if (i + 1) % self.batch_size == 0:
                self.feature_batches.append(tmp_feature_batch)
                self.label_batches.append(tmp_label_batch)
                tmp_feature_batch = []
                tmp_label_batch = []
        
        if len(tmp_feature_batch) != 0:
            for i in range(self.batch_size - len(tmp_feature_batch)):
                r = random.choice(range(len(self.data_paths)))
                tmp_feature_batch.append(self.data_paths[r])
                tmp_label_batch.append(self.label_paths[r])
            
            self.feature_batches.append(tmp_feature_batch)
            self.label_batches.append(tmp_label_batch)
        
        self.feature_batches = np.asarray(self.feature_batches)
        self.label_batches = np.asarray(self.label_batches)

    def __len__(self):
        return len(self.feature_batches)

    def __getitem__(self, idx):

        feature_batch_paths = self.feature_batches[idx]
        label_batch_paths = self.label_batches[idx]

        for i in range(len(feature_batch_paths)):
            assert feature_batch_paths[i].split("/")[-1] == label_batch_paths[i].split("/")[-1]

        feature_batch_paths = np.asarray([json.load(open(path, "r")) for path in feature_batch_paths])
        label_batch_paths = np.asarray([json.load(open(path, "r")) for path in label_batch_paths])

        return feature_batch_paths.squeeze(), to_categorical(label_batch_paths.squeeze(), 8)

if __name__ == "__main__":

    dataset = AudioEmotionDatasetV2("audio_data_emotions_images_0.1", 4, "val")

    d, l = dataset[0]
    print(d.shape)
    print(l.shape)