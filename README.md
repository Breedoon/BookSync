# BookSync Project Description

**Product:** An app allowing to synchronously read a book and listen to its audiobook, without even moving your eyes. This way, it is possible to drastically improve reading speed while preserving understanding of the information. Additionally, the app and can be used for effective multimodal language learning ([Demo](https://drive.google.com/open?id=1-egH8VuYbGKOoQP3S6fIQGy-p6saUO0U) of the idea). 

## Motivation

I've always preferred audiobooks to books, but sometimes I needed to concentrate on the book more than I could concentrate on an audiobook, so I tried both at the same time which indeed allowed me to process more information and do it faster. However, I felt like it could be even faster if my eyes didn't have to bounce around back and forth between the lines but were focused on one spot (like what Spritz/Speedreader do) while hearing the exact word I'm reading. This seemed like a tool that surely many people will find useful and yet, I found nothing good enough. Most of the apps I found used a synthesized voice (instead of a synced audiobook) and required my eyes to move between the lines (though sometimes with a blob highlighting which word was being read). I was disappointed not to find anything worthy, but I could still read a book while listening to the audiobook to achieve sufficient results.

On a separate ocasion, I tried to improve my skills in a foreign language similarly: by reading a book while listening to its audiobook. This approach seemed like a good way to learn a language since it gives exposure to both spoken and written language. However, it turned out that this time not having a dedicated application that would make it seamless to look up a word without having to manually pause the audiobook and then spend a minute trying to find where you left it off was critical, and this time I coudln't achieve a good enough result myself that easily.

I didn't get a chance even to start coding such an app because of how overwhelming the whole project seemed, but I took a machine learning course at Minerva and read articles on NLP, which gave me a general idea of how it can be approached.

## Technical Details

The product I have in mind consists of two parts: a machine learning model that pre-processes books by synchronizing them with audiobooks, and an app that utilizes those resources and conveniently displays them to a user. 

Technically, using audiobooks isn't necessary, and speech can be synthesized, which is true, but most people who regularly listen to audiobooks would readily prefer listening to a human who knows which words to emphasize and how to read dialogs, etc. instead of an even convincingly sounding robot. Yet, not all books have audiobooks, so synthesizing one out of pure text may be a necessary last resort.

The input to the model would be an audio file (e.g., imported from Audible) and a long list of words, and the output would be rouhgly a tuple of (word, start_time, end_time), which will then be utilized by the user application.

Using the current stat-of-art language recognition tools, it is possible to train a model (or use a pre-trained one) that would output a probability distribution of possible phonemes produced at any given time of a recording, e.g., feeding it the prononciation of the word "know" could result in roughly:

| Sound\\Time | 0 s | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1.0 s |
| ----------- | --- | ----- | ----- | ----- | ----- | ----- |
| n           | 0.8 | 0.6   | 0.1   | 0     | 0     | 0     |
| ng          | 0.2 | 0.2   | 0.1   | 0     | 0     | 0     |
| oh          | 0   | 0.2   | 0.5   | 0.6   | 0.3   | 0.05  |
| ah          | 0   | 0.1   | 0.3   | 0.4   | 0.3   | 0.05  |
| oo          | 0   | 0     | 0     | 0     | 0.2   | 0.9   |

Which can then be interpreted as "no", "know", or "now". Normally, speech-to-text tools rely on the context to discern among them, which is often pretty inaccurate. But comparing it to the actual text (for example, if it has only "know" in the vicinity) will almost certainly narrow it down to just one option, thus allowing us to find the exact timestamp of that word. Some errors will be made, but their frequency will likely be very low, and so it should be highly possible to achieve a near-perfect synchronization in most cases.

The app would allow the user to choose the speed and reading mode, easily look up a word, adjust the delay between sound and image, and manually resolve potential errors or conflicts between the book and the audiobook.