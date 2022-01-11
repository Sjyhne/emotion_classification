import numpy as np
import librosa

def get_actor_split(itemlist, splits):
    split = []
    for i, spl in enumerate(splits[:-1]):
        split.append(sum(splits[:i]) + spl)
    train, val, test = np.split(itemlist, [int(len(itemlist)*spl) for spl in split])
    
    return train, val, test

def get_duration_span(filepaths):

    durations = []

    for path in filepaths:
        wav, sr = librosa.load(path=path, sr=None)
        duration = librosa.get_duration(wav, sr)
        durations.append(duration)

    return min(durations), max(durations)