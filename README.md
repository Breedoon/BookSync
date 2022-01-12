# BookSync

**Table of Contents**

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
	- [Inputs](#inputs)
	- [Outputs](#outputs)
	- [Examples](#examples)


## About The Project  

The idea behind this project is to create a reader app allowing to synchronously read a book while listening to its audiobook which should significantly increase reading efficiency (speed and retention). Additionally, the app and can be used for multimodal language learning. See the [Examples](#examples) section for a demo.

## Getting Started  

1. Install [Python 3.7](https://www.python.org/downloads/release/python-379/) if not installed (newer versions might not work because DeepSpeech is stuck at a deprecated TensorFlow).
2. Install [`ffmpeg`](https://ffmpeg.org/) and make sure it is in PATH.
3. Clone the [GitHub repo](https://github.com/Breedoon/BookSync). 
4. Install the requirements from `requirements.txt`.
5. Download the [compiled model](https://yadi.sk/d/-f7U0uAcjiki2g) and put it into `assets/` (it's too large to fit onto GitHub).

## Usage  

### Inputs

The main file to run is `app.py`. Its `main()` function takes two main arguments: (1) the path to the audio file of the audiobook (of pretty much any format - handled by `ffmpeg`) and (2) the path to the `.txt` file with the content of the same book. It also takes arguments that allow to bound the contents of the files, ie specify from which part of the recording and the book to only try to sync (though the beginnings and the ends of those parts should match).

### Outputs

As this is still a proof of concept, the only output it produces is a video which displays words one-by-one in sync with the recording it plays. By, default, the location of such output will be `out/out.mp4`.

### Examples

At the bottom of `app.py` you will find a few examples of running the script for excerpts from the book of different length.

For example:

```python
main(
 audio_file='in/02.mp3',  
 txt_file='in/02.txt',  
 start_sec=11 * 60 + 33, # 11:33  
 end_sec=11 * 60 + 53, # 11:53  
 start_words='with the progressive dawn',  
 end_words='destination we did not know'  
)
```

attempts to sync a [20-second excerpt](in/in.wav) with its transcript:

> With the progressive dawn, the outlines of an immense camp became visible: long stretches of several rows of barbed wire fences; watch towers; searchlights; and long columns of ragged human figures, grey in the greyness of dawn, trekking along the straight desolate roads, to what destination we did not know. There were isolated shouts and whistles of command. We did not know their meaning. My imagination led me to see gallows with people dangling on them. I was horrified, but this was just as well, because step by step we had to become accustomed to a terrible and immense horror.

and as a result produces the following [video file](out/out.mp4):

https://user-images.githubusercontent.com/43219473/149152537-86bb166f-cff8-4fd3-b0c2-115e6f39e0ce.mp4
