import cv2
import numpy as np


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
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 4 == 0:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]

                frames.append(frame)
                frame_labels.append(np.array(label))
            
            frame_count += 1

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