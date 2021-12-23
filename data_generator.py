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

from video_utils import load_video

from tensorflow.keras.utils import Sequence

from audio_utils import read_audio, read_as_melspectrogram, normalize

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

        self.img_h = 318
        self.img_w = 128
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

        

if __name__ == "__main__":

    data = AudioEmotionDataset("audio_data_emotions", 16)

    print(data[0][0].shape)
    print(data[0][1].shape)

    print(data[0][0][:, :, :, 3])