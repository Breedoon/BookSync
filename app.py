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
from utils.fastdtw import dtw
from matplotlib import animation
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


def main():
    m = STSyncer(  # 20-second excerpt
        audio_file='in/02.mp3',
        txt_file='in/02.txt',
        start_sec=11 * 60 + 33,  # 11:33
        end_sec=11 * 60 + 53,  # 11:53
        start_words='with the progressive dawn',
        end_words='destination we did not know'
    )

    # m = STSyncer(  # 1-minute excerpt
    #     audio_file='in/02.mp3',
    #     txt_file='in/02.txt',
    #     start_sec=89,
    #     end_sec=125,
    #     start_words='while these ordinary',
    #     end_words='psychological basis'
    # )

    # m = STSyncer(  # 5-minute excerpt
    #     audio_file='in/02.mp3',
    #     txt_file='in/02.txt',
    #     start_sec=3 * 60 + 7,  # 3:07
    #     end_sec=8 * 60 + 47,  # 8:47
    #     start_words='a definite number of',
    #     end_words='lost their validity'
    # )

    # m = STSyncer('in/02.mp3', 'in/02.txt')  # full 27-minute excerpt

    m.make_csv()
    m.make_animation()


class STSyncer:
    """Speech & Text Syncer"""
    WAV_AUDIO_FILE = 'in/in.wav'

    def __init__(self, audio_file, txt_file, start_sec=None, end_sec=None, start_words=None, end_words=None):
        self.audio_file = audio_file
        self.txt_file = txt_file
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.start_words = start_words
        self.end_words = end_words

        self.path_cost = None
        self.wav_framerate = None

    def make_csv(self, save_to='out/out.csv'):
        word_ids = np.array([self.word_id_on_step_n(step) for step in range(len(self.path))])
        # indexes where next element != previous; word_starts[i] = first audio step of ith word
        word_starts = np.where(word_ids[1:] != word_ids[:-1])[0] + 1
        pd.DataFrame(word_starts, columns=['start_step']).to_csv(save_to, index_label='word_id')

    def make_animation(self, save_to='out/out.mp4'):
        print('Compiling an animation to visualize the result...')

        def animate(step):
            plt.gca().axis('off')
            word = self.word_from_id(self.word_id_on_step_n(step))
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

    def word_id_on_step_n(self, n):
        trpt_i = self.path[n][1]
        word_i = self.trpt_id_to_word_id_map[trpt_i]
        if pd.isna(word_i):  # words haven't begun yet
            word_i = -1
        else:
            word_i = int(word_i)
        return word_i

    def word_from_id(self, word_id):
        if word_id == -1:  # words haven't begun yet
            return ''
        return self.transcript[self.word_starts[word_id]: self.word_ends[word_id]]

    @property
    @lru_cache()
    def spaces_inds(self):
        return np.argwhere(self.transcript_inds == 0)[:, 0]

    @property
    @lru_cache()
    def trpt_id_to_word_id_map(self):
        ind_word_mad = pd.Series([np.nan] * len(self.transcript_inds))
        ind_word_mad[self.word_starts] = np.arange(len(self.word_starts))
        ind_word_mad = ind_word_mad.ffill()
        return ind_word_mad

    @property
    @lru_cache()
    def word_starts(self):
        """Audio indexes of beginnings of each word in transcript"""
        return self.spaces_inds[:-1] + 1

    @property
    @lru_cache()
    def word_ends(self):
        """Audio indexes right after every words ends, so that ith word is trpt[word_starts[i]: word_ends[i]]"""
        return self.spaces_inds[1:]  # - 1

    @property
    @lru_cache()
    def path(self):
        true_duration_steps = len(self.samples) // (self.wav_framerate * STEP_MS // 1000)
        shift_by_ts = len(self.logits) - true_duration_steps  # shift transcript back by 10 * 20 = 200 ms
        sub_path = np.array(self.full_path)
        sub_path[:-shift_by_ts, 1] = sub_path[shift_by_ts:, 1]
        self.path_cost = self.costs[sub_path[:, 0], self.transcript_inds[sub_path[:, 1]]]
        return sub_path

    @property
    @lru_cache()
    def full_path(self):
        print('Running DTW to map transcript onto the probability distributions...')

        def dist(mat_inds, trpt_inds):
            # remove -1s if any (when array was odd but needed to be reduced by half)
            mat_inds = mat_inds[~(mat_inds == -1)]
            trpt_inds = trpt_inds[~(trpt_inds == -1)]
            return -self.costs[mat_inds].mean(axis=0)[trpt_inds].mean()

        dist, path = fastdtw(np.arange(len(self.logits)).reshape(-1, 1),
                             self.transcript_inds.reshape(-1, 1),
                             radius=self.radius,
                             dist=dist,
                             max_approximations=1)
        return path

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
    def radius(self):
        return 200

    @property
    @lru_cache()
    def logits(self):
        print('Running DNN to infer character probability distributions of the audio file...')

        # with open("assets/logits.pickle", "rb") as f:  # for debugging: to not have to rerun DNN on the same file
        #     return pickle.load(f)

        logits = infer_character_distribution(self.wav_audio_file)
        with open("assets/logits.pickle", "wb") as f:
            pickle.dump(logits, f)

        return logits

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