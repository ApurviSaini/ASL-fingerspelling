from AlphabetPrediction import AlphabetPrediction
from Prediction.WordPrediction import WordPrediction
from hand_shape_extraction_from_frame import extractHandFrame
from Naked.toolshed.shell import execute_js, muterun_js
import cv2
import os
import time
from Utils import getCsvFromJson
from utils.Utils import getFrames


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ALPHABET_EXTRACTED_FRAMES_PATH = os.path.join(FILE_PATH, 'alphabetExtractedVideoFrames')
WORD_HANDFRAMES_PATH = os.path.join(FILE_PATH, 'wordExtractedHandFrames')
ALPHABET_TRAINING_PATH = os.path.join(FILE_PATH, 'alphabetTrainVideos')
ALPHABET_HANDFRAMES_PATH = os.path.join(FILE_PATH, 'alphabetExtractedHandFrames')
WORD_TRAINING_PATH = os.path.join(FILE_PATH, 'wordTrainVideos')
WORD_EXTRACTED_FRAMES_PATH = os.path.join(FILE_PATH, 'wordExctractedFrames')




def predictAlphabet():
    getFrames(ALPHABET_TRAINING_PATH, ALPHABET_EXTRACTED_FRAMES_PATH)
    for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
        print("Now lets create key points file for the alphabet {}".format(letter))
        fp = os.path.join(ALPHABET_EXTRACTED_FRAMES_PATH, "{}/".format(letter))
        success = execute_js('posenet.js', fp)
        if success == True:
            getCsvFromJson(letter)
            cropped_folder = os.path.join(ALPHABET_HANDFRAMES_PATH, "{}_cropped".format(letter))
            extractHandFrame(fp, cropped_folder)
    AlphabetPrediction(alphabetVideoPath=ALPHABET_TRAINING_PATH, alphabetFramePath=ALPHABET_HANDFRAMES_PATH).predict()


def predictWord():
    getFrames(WORD_TRAINING_PATH, WORD_EXTRACTED_FRAMES_PATH)
    videoFileNames = [file for file in os.listdir(WORD_TRAINING_PATH) if file.endswith('.mp4')]

    for fileName in videoFileNames:
        word = fileName.split('.')[0]
        print("Creating a keypoints file for the word {}".format(fileName.split('.')[0]))
        fp = os.path.join(WORD_EXTRACTED_FRAMES_PATH, "{}/".format(word))
        success = execute_js('posenet.js', fp)
        if success == True:
            getCsvFromJson(fp)
            cropped_folder = os.path.join(WORD_HANDFRAMES_PATH, "{}_cropped".format(word))
            extractHandFrame(fp, cropped_folder)
    WordPrediction(wordVideoPath=WORD_TRAINING_PATH,wordFramePath=WORD_HANDFRAMES_PATH,posKeyPath=WORD_EXTRACTED_FRAMES_PATH).predict()


print("Enter 1 for alphabet prediction with videos training\n")
print("Enter 2 Word Prediction with videos training\n")
option = input("Choose an option: ")
{'1':predictAlphabet,'2':predictWord}[option]()
