import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio
from utils.text import Alphabet
from tqdm import tqdm
import os


MODELS = dict(
    en='assets/model.en.tflite',
    pl='assets/model.pl.tflite',
    de='assets/model.de.tflite',
    es='assets/model.es.tflite',
)

n_input = 26
n_context = 9
n_steps = 16
n_cell_dim = 2048
WINDOW_MS = 32
STEP_MS = 20  # feature extraction window step length


def samples_to_mfccs(samples, sample_rate):
    # 16000 = default sample rate
    # 32 = default feature extraction audio window length in milliseconds
    audio_window_samples = 16000 * (WINDOW_MS / 1000)
    # 20 = default feature extraction window step length in milliseconds
    audio_step_samples = 16000 * (STEP_MS / 1000)
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=audio_window_samples,
                                                  stride=audio_step_samples,
                                                  magnitude_squared=True)

    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=n_input)
    mfccs = tf.reshape(mfccs, [-1, n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features(wav_filename):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)
    return features, features_len


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * n_context + 1
    num_channels = n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                             .reshape(window_width, num_channels, window_width * num_channels),
                             tf.float32)  # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


def infer_character_distribution(input_file_path, lang):
    """Load model from checkpoint, run inference and return most likely predicted characters"""
    inter = tf.lite.Interpreter(model_path=MODELS[lang])
    inter.allocate_tensors()

    tensors = {}
    tensor_details = inter.get_input_details() + inter.get_tensor_details() + inter.get_output_details()
    for tensor in tensor_details:
        tensors[tensor['name']] = tensor

    with tf.compat.v1.Session() as session:
        # Get input and output tensors
        features, features_len = audiofile_to_features(input_file_path)
        previous_state_c = np.zeros([1, n_cell_dim]).astype('float32')
        previous_state_h = np.zeros([1, n_cell_dim]).astype('float32')

        # Add batch dimension
        features = tf.expand_dims(features, 0)
        features_len = tf.expand_dims(features_len, 0)

        # Evaluate
        features = create_overlapping_windows(features).eval(session=session)
        features_len = features_len.eval(session=session)

        logits = list()

        # the frozen model only accepts input split to 16 step chunks,
        # if the inference was run from checkpoint instead (as in single inference in deepspeech script), this loop wouldn't be needed
        for i in tqdm(range(0, features_len[0], n_steps)):
            chunk = features[:, i:i + n_steps, :, :]
            chunk_length = chunk.shape[1];
            # pad with zeros if not enough steps (len(features) % FLAGS.n_steps != 0)
            if chunk_length < n_steps:
                chunk = np.pad(chunk,
                               (
                                   (0, 0),
                                   (0, n_steps - chunk_length),
                                   (0, 0),
                                   (0, 0)
                               ),
                               mode='constant',
                               constant_values=0)

            inter.set_tensor(tensors['input_node']['index'], chunk)
            inter.set_tensor(tensors['previous_state_c']['index'], previous_state_c)
            inter.set_tensor(tensors['previous_state_h']['index'], previous_state_h)
            inter.invoke()
            logits_step = inter.get_tensor(tensors['logits']['index'])
            previous_state_c = inter.get_tensor(tensors['new_state_c']['index'])
            previous_state_h = inter.get_tensor(tensors['new_state_h']['index'])

            logits.append(logits_step)

    logits = np.squeeze(np.concatenate(logits))

    return logits