from __future__ import print_function

import os
import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf

import vggish.vggish_input as vggish_input
import vggish.vggish_params as vggish_params
import vggish.vggish_postprocess as vggish_postprocess
import vggish.vggish_slim as vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def create_and_save_features(wav_filepaths, datatype, dst_path):

    pca_params_path = "vggish/vggish_pca_params.npz"
    checkpoint_path = "vggish/vggish_model.ckpt"

    dst_path = os.path.join(dst_path, datatype)

    os.makedirs(dst_path)

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

        for audiotype, filepaths in wav_filepaths.items():
            
            print("Audio type:", audiotype, "| filepaths:", len(filepaths))

            for filepath in filepaths:
                

            
                batch = vggish_input.wavfile_to_examples(filepath)
                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                            feed_dict={features_tensor: batch})

                postprocessed_batch = pproc.postprocess(embedding_batch)

                for idx, postprocessed in enumerate(postprocessed_batch):
                    new_filename = filepath.split("/")[-1].split(".")[0] + f"_{idx}.npy"
                    final_filepath = os.path.join(dst_path, new_filename)
                    print("finalfilepath:", final_filepath)

                    np.save(final_filepath, postprocessed)

                # TODO: Save the features to some dir with the filename
                # This makes me able to retrieve the label based on the
                # file name and therefore get the label when the npy
                # is read from the file. Should also store them separately
                # as I want to train on them separately, not in batches



def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.

  wav_file = "/home/sj/emotion_classification/datasets/CREMA-D/AudioWAV/1001_IEO_SAD_HI.wav"

  examples_batch = vggish_input.wavfile_to_examples(wav_file)
  print(examples_batch)

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  # If needed, prepare a record writer to store the postprocessed embeddings.
  writer = tf.python_io.TFRecordWriter(
      FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    print(embedding_batch.shape)
    print("embedding_batch.shape:", embedding_batch.shape)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print("postprocessed_batch:", postprocessed_batch.shape)
	
    # Write the postprocessed embeddings as a SequenceExample, in a similar
    # format as the features released in AudioSet. Each row of the batch of
    # embeddings corresponds to roughly a second of audio (96 10ms frames), and
    # the rows are written as a sequence of bytes-valued features, where each
    # feature value contains the 128 bytes of the whitened quantized embedding.
    seq_example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[embedding.tobytes()]))
                            for embedding in postprocessed_batch
                        ]
                    )
            }
        )
    )
    print("Seq example:", seq_example)
    if writer:
      writer.write(seq_example.SerializeToString())

  if writer:
    writer.close()

if __name__ == '__main__':
  create_features("/home/sj/emotion_classification/datasets/CREMA-D/AudioWAV/1001_IEO_HAP_LO.wav")