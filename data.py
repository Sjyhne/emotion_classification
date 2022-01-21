from cv2 import merge
import numpy as np
import matplotlib.pyplot as plt
from dataset_loaders.cremad import get_cremad_data, get_cremad_label
import librosa
from dataset_loaders.emodb import get_emodb_data, get_emodb_label
from dataset_loaders.shemo import get_shemo_data, get_shemo_label
from dataset_loaders.emovo import get_emovo_data, get_emovo_label

from audio_preprocessing import create_spectrogram_and_save_images, load_wav, get_framed_mel_spectrograms

import soundfile
import tensorflow.compat.v1 as tf

import vggish.vggish_input as vggish_input
import vggish.vggish_params as vggish_params
import vggish.vggish_postprocess as vggish_postprocess
import vggish.vggish_slim as vggish_slim

import random
import shutil
import os

EMOTION_MAP = {
    "anger": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}

def get_class_diversity(filepaths):
    class_diversity = {v: 0 for v in EMOTION_MAP.values()}
    for filepath in filepaths:
        label = get_label(filepath)
        if label != "N/A":
            class_diversity[label] += 1
    
    return class_diversity

def combine_datatypes(data):
    result_data = []
    for d in data:
        result_data.extend(d)

    return result_data

def get_percentage_of_data(data, data_percentage):
    new_data = []
    for d in data:
        if len(new_data) < int(len(data) * data_percentage):
            new_data.append(d)
        else:
            break
    return new_data

def merge_datasets(data_percentage=1.0):

    train_emovo, val_emovo, test_emovo = get_emovo_data()
    train_emodb, val_emodb, test_emodb = get_emodb_data()
    train_shemo, val_shemo, test_shemo = get_shemo_data()
    train_cremad, val_cremad, test_cremad = get_cremad_data()

    train_paths = combine_datatypes([train_emovo, train_emodb, train_shemo, train_cremad])
    val_paths = combine_datatypes([val_emovo, val_emodb, val_shemo, val_cremad])
    test_paths = combine_datatypes([test_emovo, test_emodb, test_shemo, test_cremad])

    print("Nr of audio-files (train, val, test):", len(train_paths), len(val_paths), len(test_paths))

    if data_percentage != 1.0:
        random.shuffle(train_paths)
        random.shuffle(val_paths)
        random.shuffle(test_paths)

        train_paths = get_percentage_of_data(train_paths, data_percentage)
        val_paths = get_percentage_of_data(val_paths, data_percentage)
        test_paths = get_percentage_of_data(test_paths, data_percentage)
    
        print("Nr of audio-files (train, val, test) after split:", len(train_paths), len(val_paths), len(test_paths))
    
    return train_paths, val_paths, test_paths

def create_and_save_features(wav_filepaths, dst_path, datatype):

    pca_params_path = "vggish/vggish_pca_params.npz"
    checkpoint_path = "vggish/vggish_model.ckpt"

    dst_path = os.path.join(dst_path, datatype)

    print("dst_path:", dst_path)

    pproc = vggish_postprocess.Postprocessor(pca_params_path)

    with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        for filepath in wav_filepaths:
            
            batch = vggish_input.wavfile_to_examples(filepath)
            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                        feed_dict={features_tensor: batch})

            postprocessed_batch = pproc.postprocess(embedding_batch)

            for idx, postprocessed in enumerate(postprocessed_batch):
                dataset = filepath.split("/")[1]
                new_filename = filepath.split("/")[-1].split(".")[0] + f"_{dataset}_{idx}.npy"
                final_filepath = os.path.join(dst_path, new_filename)

                print("final_filepath:", final_filepath)

                np.save(final_filepath, postprocessed)

def create_vggish_features_dataset(target_dir, split_val=1.0):

    if split_val != 1.0:
            target_dir = target_dir + "_" + str(split_val)

    datatypes = ["train", "val", "test"]

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    
    for datatype in datatypes:
        os.mkdir(os.path.join(target_dir, datatype))
    
    tr, va, te = merge_datasets(split_val)

    print("Training class diversity:", get_class_diversity(tr))
    print("Val class diversity:", get_class_diversity(va))
    print("Test class diversity:", get_class_diversity(te))

    create_and_save_features(tr, target_dir, "train")
    create_and_save_features(va, target_dir, "val")
    create_and_save_features(te, target_dir, "test")
    
    

def create_spectogram_dataset(target_dir, split_val=1.0):

    if split_val != 1.0:
        target_dir = target_dir + "_" + str(split_val)

    datatypes = ["train", "val", "test"]

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    
    for datatype in datatypes:
        os.mkdir(os.path.join(target_dir, datatype))
    
    tr, va, te = merge_datasets(split_val)

    print("Training class diversity:", get_class_diversity(tr))
    print("Val class diversity:", get_class_diversity(va))
    print("Test class diversity:", get_class_diversity(te))

    create_spectrogram_and_save_images(tr, "train", target_dir)
    create_spectrogram_and_save_images(va, "val", target_dir)
    create_spectrogram_and_save_images(te, "test", target_dir)
    

def get_label(path):

    if "EMOVO" in path:
        label = get_emovo_label(path)
    elif "ShEMO" in path:
        label = get_shemo_label(path)
    elif "CREMA-D" in path:
        label = get_cremad_label(path)
    else:
        label = get_emodb_label(path)
    return EMOTION_MAP[label]


if __name__ == "__main__":
    create_vggish_features_dataset("vggish_features", 1.0)


