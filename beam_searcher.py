import re
import numpy as np


def get_transcript_char_mapping(chars: str, transcript: str):
    """For each character of the alphabet, returns indices where it occurs in transcript"""
    mapping = [[]] * len(chars)
    for i, char in enumerate(chars):
        mapping[i] = [i.start() for i in re.finditer(char, transcript)]
    return mapping


class Path:
    def __init__(self, score=1, path=(), time=(), transcript_covered=None, transcript=None):
        """
        Path object of walking through the transcript and mapping audio onto it
        :param float score: Score of this path
        :param tuple path: Indices of transcript in the path
        :param tuple time: Timesteps corresponding to the indices in the path above (len(time) == len(path))
        :param set transcript_covered: set of unique indices of transcript covered
        :param str transcript: optional transcript to translate indices into characters for visualization
        """
        self.score = score
        self.path = tuple(path)
        self.time = tuple(time)
        self.transcript = transcript

        # if transcript_covered is None

    def __str__(self):
        if self.transcript is not None:  # show current transcription
            path = ''.join([self.transcript[i] for i in self.path])
        else:
            path = self.path
        return f"Path(score={round(self.score, 5)}, path={path})"

    def __repr__(self):
        return str(self)

    def score_of_next_at(self, base_prob, at_ind):
        """
        Cost of the next character to be added to the path is from `at_ind` index of the transcript
        i.e., if the path is empty should be 100%;
        if path is [0, 0, 1, 1, 2] then if `at_ind` is 3, the probability should be high
        but if `at_ind` is 100, the probability should be low
        """
        if len(self.path) == 0:  # new path
            return self.score + base_prob
        if at_ind < self.path[-1]:
            return 0  # cannot come before existing path
        # farther away from last -> lower probability
        return self.score * (1 + self.loc_coef(at_ind - self.path[-1]) * base_prob)

    @staticmethod
    def loc_coef(dist, coef=0.1, lowest_val=0.1):
        """Converts distance between two chars to coeffecient to the score"""
        scores = [0.5, 1, 0.2]  # scores[dist] - multiplier of score if the new candidate is dist away
        if dist >= len(scores):
            return -0.01 * dist
        return scores[dist]

    @staticmethod
    def get_top_paths(paths_candidates, top_n=25, sort=True):
        """
        Returns top `top_n` paths based on their probability

        :param paths_candidates: list of `Path`s
        :param top_n: The number of most probably paths to retrieve
        :param sort: if true, will sort the output (is a bit slower); if False, the order is arbitrary
        :return:
        """
        if len(paths_candidates) < top_n:  # fewer paths given than N
            if sort:
                return sorted(paths_candidates, key=lambda p: p.score, reverse=True)  # sort all by probability
            else:
                return paths_candidates

        scores = np.array([path.score for path in paths_candidates])

        if sort:
            sort_inds = np.arange(top_n)
        else:
            sort_inds = top_n

        highest_inds = np.argpartition(-scores, sort_inds)[:top_n]
        return [paths_candidates[i] for i in highest_inds]


def infer_transcript_timing(mat: np.ndarray, chars: str, beam_width=25, transcript=None, **kwargs):
    char_map = get_transcript_char_mapping(chars, transcript)

    last_paths = [Path(transcript=transcript)]

    for t in range(len(mat)):
        best_paths = Path.get_top_paths(last_paths, top_n=beam_width)  # do not sort b/c doesn't matter  sort=False
        new_paths = [Path(score=t / 10, transcript=transcript)]  # initialize with an empty path
        for path in best_paths:
            for c in range(len(chars)):
                base_prob = mat[t, c]
                for i in char_map[c]:
                    if len(path.path) > 0 and i < path.path[-1]:
                        continue
                    new_score = path.score_of_next_at(base_prob, i, **kwargs)
                    new_paths.append(Path(score=new_score,
                                          path=path.path + (i,),
                                          time=path.time + (t,),
                                          transcript=transcript))
            new_paths.append(Path(path.score, path.path, path.time, transcript=transcript))  # try skipping current step
        last_paths = new_paths
    return last_paths
