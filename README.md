# joint-hands-gestures
A algorithm that detects and tracks hands and regcognizes joint hand gestures. Can also cut video automatically and has a transcription helper.

## What is it?

The repository consists of three independently useable files.

`cut_video.py` can be used to cut a video based on a reference frame. It has been used to extract a single camera perspective out of a video of a political speech. Note that it creates an output file without an audio track. The audio track should be in the same folder with the same name as the video, then it will get cut accordingly to the video.

`analyse_video.py` consists of two main parts. The first part is an automated tracking of hands on a per frame basis. This creates a numpy array of the position of each hand for each frame. It also tracks when both hands are joint. In the second part all joint hand gestures will be cut into seperate video (and audio files if available).

`transcribe_audio.py` can be used to transcribe a large number of small audio files. It uses regex to extract an id (or something similar) out of the file name which then can used to match the transcribed text to the audio.

## Why

Created for a final exam in the course Cognitive Science 2 at the University of Copenhagen.
