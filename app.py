import numpy as np
import pandas as pd
from pandas import Timedelta as td
import matplotlib.pyplot as plt
import os
from utils.audio_to_logits import infer_character_distribution, alphabet, STEP_MS
from utils.text import preprocess_transcript, preprocess_audio
import pickle
import struct
import wave
from functools import lru_cache
from utils.fastdtw import fastdtw
from matplotlib import animation
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


def main():
    m = Mapper('in/02.mp3', 'in/02.txt', 89, 125, 'while these ordinary', 'psychological basis')
    m.plot_fit()
    m.make_animation()


class Mapper:
    WAV_AUDIO_FILE = 'in/in.wav'

    def __init__(self, audio_file, txt_file, start_sec=None, end_sec=None, start_words=None, end_words=None):

        self.audio_file = audio_file
        self.txt_file = txt_file
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.start_words = start_words
        self.end_words = end_words
        self.radius = 50

        self.path_cost = None
        self.wav_framerate = None

    def make_animation(self, save_to='out/out.mp4'):
        spaces = np.argwhere(self.transcript_inds == 0)[:, 0]
        word_starts = spaces[:-1] + 1
        word_ends = spaces[1:]  # - 1
        np.vstack([word_starts, word_ends])

        ind_word_mad = pd.Series([np.nan] * len(self.transcript_inds))
        ind_word_mad[word_starts] = np.arange(len(word_starts))
        ind_word_mad = ind_word_mad.ffill()

        # sub_path_cost[word_starts[55]:word_ends[55]].mean()

        def animate(step):
            plt.gca().axis('off')
            trpt_i = self.path[step][1]
            word_i = ind_word_mad[trpt_i]
            if pd.isna(word_i):  # words haven't began yet
                word = ''
            else:
                word_i = int(word_i)
                word = self.transcript[word_starts[word_i]: word_ends[word_i]]
            return [plt.annotate(word, xy=(0.5, 0.5), fontsize=26, ha='center', va='center')]

        class tqdmlist(list):
            """Custom class to show the progress of the animation creation with tqdm (to not bother with progress callback)"""

            def __iter__(self):
                return self.tqdm_iter(super(tqdmlist, self).__iter__())

            @staticmethod
            def tqdm_iter(itr):
                for it in tqdm(list(itr)):  # to list to show progress
                    yield it

        fig = plt.figure()

        frames = tqdmlist([animate(step) for step in tqdm(range(len(self.path)))])
        anim = animation.ArtistAnimation(fig, frames, interval=STEP_MS, blit=True)

        animation_output_file = 'out/animation.mp4'

        # If animation file already exists
        if os.path.exists(animation_output_file):
            os.remove(animation_output_file)

        anim.save('out/animation.mp4', writer='ffmpeg')

        if os.path.exists(save_to):
            os.remove(save_to)

        cmd = f"ffmpeg -i {animation_output_file} -i {self.wav_audio_file} -c:v copy -c:a aac {save_to}"
        os.system(cmd)

        # Remove animation file
        os.remove(animation_output_file)

    def plot_fit(self, save_to='out/fit.png'):
        fig, [ax3, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=(len(self.logits) / 20, 5),
                                            gridspec_kw=dict(height_ratios=[1, 1, 4]))

        t = np.linspace(0, len(self.samples) / self.wav_framerate, len(self.samples))
        ax2.plot(t, self.samples, color='pink', lw=1)

        next_j = self.path[1][1]
        for i, (rec_i, trpt_i) in enumerate(self.path[:-1]):
            next_j = self.path[i + 1][1]
            if rec_i != next_j:
                ax1.annotate(self.transcript[trpt_i], xy=(rec_i * STEP_MS / 1000, 0.5))
            # print(i, j, transcript[i])

        ax3.plot(np.linspace(0, len(self.costs) * STEP_MS / 1000, num=len(self.costs)), self.path_cost, color='k')
        ax3.set_yscale('log')
        ax1.axes.get_xaxis().set_visible(False)

        ax3.set_xlim(0, t.max())
        ax2.set_xlim(0, t.max())
        ax1.set_xlim(0, t.max())

        ax2.set_xlabel('Time (s)')
        # hide axes for text
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        plt.gcf().savefig(save_to)
        plt.show()

    @property
    @lru_cache()
    def path(self):
        shift_by_ts = len(self.logits) - len(self.step_samples)  # shift transcript back by 10 * 20 = 200 ms
        sub_path = np.array(self.full_path)
        sub_path[:-shift_by_ts, 1] = sub_path[shift_by_ts:, 1]
        self.path_cost = self.costs[sub_path[:, 0], self.transcript_inds[sub_path[:, 1]]]
        return sub_path

    @property
    @lru_cache()
    def full_path(self):
        sliding_step = 50
        overlap_n = 1
        windows = np.array([np.arange(0, len(self.logits), sliding_step // overlap_n)[:-overlap_n],  # from
                              np.arange(0, len(self.logits), sliding_step // overlap_n)[overlap_n:]]).T  # to
        windows = np.vstack([windows, [windows[-1, -1], len(self.logits)]])  # add last steps that didn't fit in window
        trpt_start_ind = 0
        paths = []
        for frm, to in tqdm(windows):
            costs = self.costs[frm:].copy()
            costs[to:] = 0
            _, window_path = fastdtw(np.arange(len(costs)), self.transcript_inds[trpt_start_ind:], radius=20,
                                     dist=lambda mat_i, trpt_i: -costs[int(mat_i), int(trpt_i)])
            paths.append(np.array(window_path[:sliding_step]) + [frm, trpt_start_ind])
            trpt_start_ind = trpt_start_ind + window_path[to - frm - 1][1]  # - 1

        return np.vstack(paths)

    @property
    @lru_cache()
    def transcript_inds(self):
        trpt_ind_map = pd.Series(np.arange(len(self.alph)), index=self.alph)

        return trpt_ind_map[list(self.transcript)].values

    @property
    @lru_cache()
    def costs(self):
        start_t, end_t = 0, len(self.logits)

        probs = self.logits.copy()[start_t:end_t, :]  # the progressive dawn
        probs[:, 0] += probs[:, -1]  # mix spaces and blanks (they seem to be mixed up anyway)

        # If the model inserts a blank or a space after a character,
        # prolong that character's probability so that DTW simply repeats it,
        # instead of stopping the thread, for example:
        #   blank  d    a    w        blank  d    a    w
        #     0    0.8  0    0          0    0.8  0    0
        #     0.5  0.1  0    0          0.5  0.5  0    0
        #     0.8  0    0    0    ->    0.8  0.4  0    0
        #     0.1  0    0.7  0          0.1  0    0.7  0
        #     0    0    0    0.7        0    0    0    0.7
        for i in range(1, len(probs)):
            probs[i, 1:-1] += probs[i, 0] * probs[i - 1, 1:-1]

        return probs

    @property
    @lru_cache()
    def transcript(self):
        return preprocess_transcript(self.txt_file, self.start_words, self.end_words)

    @property
    @lru_cache()
    def logits(self):
        with open("assets/logits.pickle", "rb") as f:
            return pickle.load(f)
        # return infer_character_distribution(self.wav_audio_file)  TODO: uncomment

    @property
    @lru_cache()
    def samples(self):
        def read_samples(wave_file, nb_frames):
            frame_data = wave_file.readframes(nb_frames)
            if frame_data:
                sample_width = wave_file.getsampwidth()
                nb_samples = len(frame_data) // sample_width
                format = {1: "%db", 2: "<%dh", 4: "<%dl"}[sample_width] % nb_samples
                return struct.unpack(format, frame_data)
            else:
                return ()

        w = wave.open(self.wav_audio_file, 'r')

        self.wav_framerate = w.getframerate()

        samples = np.array(read_samples(w, w.getnframes()))

        return samples

    @property
    @lru_cache()
    def step_samples(self):
        return self.samples.reshape((-1, self.wav_framerate * STEP_MS // 1000))

    @property
    @lru_cache()
    def alph(self):
        return np.array(list(alphabet._str_to_label.keys()) + ['-'])

    @property
    @lru_cache()
    def wav_audio_file(self):
        return preprocess_audio(self.audio_file, self.WAV_AUDIO_FILE, self.start_sec, self.end_sec)


if __name__ == '__main__':
    main()
