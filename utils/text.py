from __future__ import absolute_import, division, print_function

import os

import codecs
import numpy as np
import re
import struct
from unidecode import unidecode
from num2words import num2words

from six.moves import range

ALPHABETS = dict(
    en='assets/alphabet.en.txt',
    pl='assets/alphabet.pl.txt',
    de='assets/alphabet.de.txt',
    es='assets/alphabet.es.txt',
)

class Alphabet(object):
    def __init__(self, lang='en'):
        self.lang = lang
        self._config_file = ALPHABETS[lang]
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 0
        with codecs.open(self._config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str[self._size] = line[:-1]  # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1

    def _string_from_label(self, label):
        return self._label_to_str[label]

    def _label_from_string(self, string):
        try:
            return self._str_to_label[string]
        except KeyError as e:
            raise KeyError(
                'ERROR: Your transcripts contain characters (e.g. \'{}\') which do not occur in data/alphabet.txt! Use ' \
                'util/check_characters.py to see what characters are in your [train,dev,test].csv transcripts, and ' \
                'then add all these to data/alphabet.txt.'.format(string)
            ).with_traceback(e.__traceback__)

    def has_char(self, char):
        return char in self._str_to_label

    def encode(self, string):
        res = []
        for char in string:
            res.append(self._label_from_string(char))
        return res

    def decode(self, labels):
        res = ''
        for label in labels:
            res += self._string_from_label(label)
        return res

    def serialize(self):
        # Serialization format is a sequence of (key, value) pairs, where key is
        # a uint16_t and value is a uint16_t length followed by `length` UTF-8
        # encoded bytes with the label.
        res = bytearray()

        # We start by writing the number of pairs in the buffer as uint16_t.
        res += struct.pack('<H', self._size)
        for key, value in self._label_to_str.items():
            value = value.encode('utf-8')
            # struct.pack only takes fixed length strings/buffers, so we have to
            # construct the correct format string with the length of the encoded
            # label.
            res += struct.pack('<HH{}s'.format(len(value)), key, len(value), value)
        return bytes(res)

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


class UTF8Alphabet(object):
    @staticmethod
    def _string_from_label(_):
        assert False

    @staticmethod
    def _label_from_string(_):
        assert False

    @staticmethod
    def encode(string):
        # 0 never happens in the data, so we can shift values by one, use 255 for
        # the CTC blank, and keep the alphabet size = 256
        return np.frombuffer(string.encode('utf-8'), np.uint8).astype(np.int32) - 1

    @staticmethod
    def decode(labels):
        # And here we need to shift back up
        return bytes(np.asarray(labels, np.uint8) + 1).decode('utf-8', errors='replace')

    @staticmethod
    def size():
        return 255

    @staticmethod
    def serialize():
        res = bytearray()
        res += struct.pack('<h', 255)
        for i in range(255):
            # Note that we also shift back up in the mapping constructed here
            # so that the native client sees the correct byte values when decoding.
            res += struct.pack('<hh1s', i, 1, bytes([i + 1]))
        return bytes(res)

    @staticmethod
    def deserialize(buf):
        size = struct.unpack('<I', buf)[0]
        assert size == 255
        return UTF8Alphabet()

    @staticmethod
    def config_file():
        return ''


def text_to_char_array(series, alphabet):
    r"""
    Given a Pandas Series containing transcript string, map characters to
    integers and return a numpy array representing the processed string.
    """
    try:
        transcript = np.asarray(alphabet.encode(series['transcript']))
        if len(transcript) == 0:
            raise ValueError(
                'While processing: {}\nFound an empty transcript! You must include a transcript for all training data.'.format(
                    series['wav_filename']))
        return transcript
    except KeyError as e:
        # Provide the row context (especially wav_filename) for alphabet errors
        raise ValueError('While processing: {}\n{}'.format(series['wav_filename'], e))


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None


def preprocess_transcript_words(words: list, alphabet=Alphabet()):
    """
    Prepares the given list of words to be used internally: ['Wørds', 'of', 'text!'] -> ' words of text '
    This includes:
    1. Converting non-ascii to closest ascii characters (except those that are part of the alphabet)
    2. Doing basic conversion of numbers to letters
    3. Removing all non-alphabet characters that didn't get processed away in previous two steps
    4. Making all characters lowercase
    5. Padding the resulting string with spaces from each side

    """
    text = '\n'.join(words)  # convert words to new-line separated text for easier processing

    chars = ''.join(alphabet._str_to_label).strip()  # alphabet as one string without space

    # try to uni-decode non-ascii to ascii to not just throw them away (eg, naïve -> naive)
    # find non-ascii characters that are part of alphabet and need to stay
    non_ascii_letters = [letter for letter in alphabet._str_to_label if unidecode(letter) != letter]
    if len(non_ascii_letters) > 0:
        # split text into parts that can and cannot be unidecoded, unidecode, the former, and join all back together
        # 'βåñøś' -> ['βå', 'ñ', 'øś'] -> ['ba', 'ñ', 'os'] -> 'baños'
        sep = '([' + ''.join(non_ascii_letters) + '])+'  # separate by non-ascii chars, eg: '([ñ])+'
        chunks = re.split(sep, text)  # split and preserve separators (non-ascii chars)
        for i in range(0, len(chunks), 2):  # odd are always separators
            chunks[i] = unidecode(chunks[i])  # so only unidecode even ones
        text = ''.join(chunks)
    else:  # if all letters are ascii (English), just unidecode the whole text
        text = unidecode(text)

    text = text.strip().lower()

    # do basic number preprocessing, '1921' -> 'nineteentwentyfive' to maintain them as one word
    for i in range(1000, -1, -1):
        text = text.replace(str(i), num2words(i, to='year', lang=alphabet.lang))

    text = re.compile(fr"[^{chars}\n']").sub('', text)  # remove all non-alphabet characters that didn't get adapted
    text = re.compile(r'\n+').sub('\n', text)
    text = text.replace('\n', ' ').strip()
    return ' ' + text + ' '


def preprocess_transcript(txt_file, alphabet=Alphabet(), start_words=None, end_words=None):
    """
    Returns a transcript string consiting of
    :param txt_file: path to the .txt file where the original text is stored.
    :param Alphabet alphabet: an Alphabet object specifying characters allowd in the alphabet (English by default)
    :param str start_words: first words to be included in the transcript from the original text, lowercase
    :param str end_words: last words to be included, also lowercase separated by spaces

    >>> preprocess_transcript('Lorem ipsum dolor sit amet, consectetur adipiscing',
    ...                       start_words='ipsum', end_words='amet consectetur')
    ' ipsum dolor sit amet consectetur '
    """

    with open(txt_file, 'r') as f:
        text = f.read()
    transcript = preprocess_transcript_words(text.split(), alphabet)

    start_i = transcript.find(start_words) - 1 if start_words is not None else 0
    end_i = transcript.find(end_words) + len(end_words) if end_words is not None else -1

    return transcript[start_i:end_i] + ' '


def preprocess_audio(in_file, out_file='in/in.wav', start_sec=None, end_sec=None):
    # Convert audio to correct format in a temp .wav file
    if os.path.exists(out_file):  # remove if temp wav file already exists
        os.remove(out_file)

    cmd = f"""ffmpeg -i "{in_file}" -acodec pcm_s16le -ac 1 -ar 16000"""

    if start_sec is not None:
        cmd += f""" -ss {start_sec}"""
    if end_sec is not None:
        cmd += f""" -to {end_sec}"""

    cmd += f""" {out_file}"""

    os.system(cmd)

    return out_file
