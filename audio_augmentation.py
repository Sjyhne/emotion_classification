import librosa
import numpy as np
import cv2 as cv

import os
import random

def time_shift(wav):
    start_ = int(np.random.uniform(-6000, 6000))
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
    
    wav_time_shift = wav_time_shift.astype(np.float32)
    
    return wav_time_shift

def speed_tuning(wav):
    speed_rate = np.random.uniform(0.7, 1.3)
    wav_speed_tune = cv.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

    wav_speed_tune = wav_speed_tune.astype(np.float32)

    return wav_speed_tune

