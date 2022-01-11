import tensorflow as tf
import librosa
import numpy as np
from tqdm import tqdm
import os

SEQ_LEN = 4
OVERLAP = 2

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

def create_spectrogram_and_save_images(data, datatype, target_dir):
    print("Creating image dataset V2")
    print(f"Moving over {datatype} data..")
    for _, path in tqdm(enumerate(data), total=len(data), desc=datatype):
        dst_path = os.path.join(target_dir, datatype, path.split("/")[-1].split(".")[0] + ".npy")
        if not os.path.exists(dst_path):
            wav, sr = load_wav(path)
            framed_log_mels = get_framed_mel_spectrograms(wav, sr)
            np_log_mels = np.asarray(framed_log_mels)
            n = SEQ_LEN
            m = OVERLAP
            chunked_mels = []
            for i in range(0, len(np_log_mels), n - m):
                tmp = np_log_mels[i:i+n]
                if len(tmp) < 4:
                    for i in range(SEQ_LEN - len(tmp)):
                        tmp = np.append(tmp, np.zeros((1, tmp[0].shape[0], tmp[0].shape[1], tmp[0].shape[2])), 0)
                    break
                chunked_mels.append(tmp)
            for idx, tmp in enumerate(chunked_mels):
                assert len(tmp) == SEQ_LEN
                tmp = np.asarray(tmp)
                dataset = path.split("/")[1]
                p = dst_path.split(".")
                p = ".".join(p[:-1]) + f"_{idx}_{dataset}." + p[-1]
                np.save(p, tmp)

def get_framed_mel_spectrograms(wav, sr=22050, mel_bins=128):
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
    num_mel_bins = mel_bins
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

def load_wav(file_path):
    wav, sr = librosa.load(file_path, mono=True)

    pre_emp = 0.97

    wav = np.append(wav[0], wav[1:] - pre_emp * wav[:-1])

    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    sr = tf.convert_to_tensor(sr, dtype=tf.float32)

    return wav, sr