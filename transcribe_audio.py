# Load the required library
import glob
import re
import os
import pygame
from pygame import mixer  
import msvcrt as ms
import sys
import numpy as np

# read all source files that have the following pattern
source_files = glob.glob("./**/RG_*_both_hands_*.mp4")

# init pygame audio player
mixer.init()

# if we want to replay the audio we can do it with this function
def replay():
    mixer.music.play()
    return

# declare text as global
text = ""

# recursive function that handles the text input
def wait():
    global text
    # read the key but as utf8 char so we can use umlauts
    key = ms.getwch()

    # first case: user is done, we save the text and continue
    if key.encode() == b'\r':
        return text + "\n"
    # second case: user wants to listen to the audio again, dont save the key
    elif key == '-':
        mixer.music.play()
        wait()
    # third case: backspace
    elif key.encode() == b'\x08':
        text = text[:-1]
        sys.stdout.flush()
        sys.stdout.write("\r {:<100}".format(text))
        wait()
    # forth case, Ctrl + C --> EXIT
    elif key.encode() == b'\x03':
    	exit()
	# all other keys just get written down
    else:
        text += key
        sys.stdout.flush()
        sys.stdout.write("\r {:<100}".format(text))
        wait()

# iterate over all found files
for i,path in enumerate(source_files):
	# print file
    print('\n', str(i) + "/" + str(len(source_files)), path)
    # match path with regex to make sure that we ahve a correct path
    m = re.match('\.\\\\RG_(\d{4}_\d{2}_\d{2}_?2?)\\\\RG_\d{4}_\d{2}_\d{2}(?:_2)?_both_hands_(\d+)\.mp4', path)
    date = m.group(1)
    fid = m.group(2)

    # check if transcriptions already exist
    trans_path = './RG_' + date + '/transcription.txt'
    if os.path.isfile(trans_path): 
        with open(trans_path, 'r') as file:
            lines = [re.match('^(\d+);.*', line) for line in file]
            ids = [l.group(1) for l in lines]
            if(fid in ids):
                continue

    # load audio and wait for user input
    audio_path = str.split(path, ".")
    audio_path[-1] = "mp3"
    audio_path = ".".join(audio_path)
    audio_path = os.path.abspath(audio_path)
    mixer.music.load(audio_path)
    mixer.music.play()
    wait()
    
    # when the user is done, write the input in the file
    with open('./RG_' + date + '/transcription.txt', 'a+') as file:
        file.write(str(fid) + ";" + text + "\n")
        text = ""

