import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf

import math
import random

from tqdm import tqdm

from tensorflow.keras.utils import Sequence

IMG_SIZE = 224
BATCH_SIZE = 8

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_labels = []

    label = int(path.split("/")[-1].split("-")[2].strip("0")) - 1
    tmp_label = [0 for i in range(8)]
    tmp_label[label] = 1
    label = tmp_label

    batches = []
    labels = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            
            frames.append(frame)
            frame_labels.append(np.array(label))

            if len(frames) == BATCH_SIZE:
                batches.append(np.array(frames))
                labels.append(np.array(frame_labels))
                frames = []
                frame_labels = []

    finally:
        cap.release()
        while len(frames) < BATCH_SIZE:
            frames.append(np.full((IMG_SIZE, IMG_SIZE, 3), 255))
            frame_labels.append(np.array(label))
        batches.append(np.array(frames))
        labels.append(np.array(frame_labels))
    
    
    return np.array(batches), np.array(labels)

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


if __name__ == "__main__":

    train_data = EmotionDataset("data_emotions_0.1", "train")

    print(len(train_data))

    for d, l in iter(train_data):
        print(d.shape, l.shape)
        print(d[0], l[0])
        exit()
    

    #for file in os.listdir("data_emotions/happy/train"):
    #    batches = load_video("data_emotions/happy/train/" + file)
    #    print(batches.shape)