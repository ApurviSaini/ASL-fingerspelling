import os
from AlphabetPredictionUtils import predictWordsFromFrames
from statistics import mode
from pandas import DataFrame
from sklearn.metrics import classification_report
import time
import pandas as pd


class WordPrediction:

    def __init__(self,wordVideoPath='',wordFramePath='',posKeyPath = '') :
        self.wordVideoPath = wordVideoPath
        self.wordFramePath = wordFramePath
        self.posKeyPath = posKeyPath

    def predict(self):
        if not os.path.exists(self.wordFramePath):
            os.makedirs(self.wordFramePath)
        pred_array = []

        video_list = [file for file in os.listdir(self.wordVideoPath) if file.endswith('.mp4')]

        for video_name in video_list:
            if video_name == '.DS_Store':
                continue
            print("Running for " + video_name)
            word_video_name = video_name.split('.')[0]
            video_name_path = "{}_Cropped".format(word_video_name)
            file_path = os.path.join(self.wordVideoPath, video_name)
            posKey = pd.read_csv(os.path.join(self.posKeyPath, word_video_name, 'key_points.csv'))
            rightWrist = posKey.rightWrist_x
            rightArm = posKey.rightWrist_y
            leftWrist = posKey.leftWrist_x
            leftArm = posKey.leftWrist_y
            word = []
            till = 0
            start = 0
            for i in range(len(rightWrist)):
                if ((i != len(rightWrist) - 1) and (
                        (abs(leftWrist[i + 1] - leftWrist[i]) > 8.5) or (abs(leftArm[i + 1] - leftArm[i]) > 8.5))):
                    till = i
                    test_data = os.path.join(self.wordFramePath, video_name_path)
                    pred = predictWordsFromFrames(test_data, start, till)
                    start = till
                    try:
                        prediction = mode(pred)
                    except:
                        prediction = ''
                    word.append(prediction)
                if (i == len(rightWrist) - 1):
                    start = till
                    till = i
                    test_data = os.path.join(self.wordFramePath, video_name_path)
                    pred = predictWordsFromFrames(test_data, start, till)
                    try:
                        prediction = mode(pred)
                    except:
                        prediction = ''
                    word.append(prediction)

            gold_label = video_name[0:3]
            print("\nSelection of Frame is Done\n")
            print("\nPredicting alphabets from frames extracted.")
            for i in range(0, 6):
                if i == 3:
                    print("generating keypoint timeseries for the word from posenet.csv")
                print("-")
                time.sleep(1)
            finalword = []
            prevchar = ''
            for i in range(0, len(word)):
                if (prevchar != word[i]):
                    finalword.append(word[i])
                prevchar = word[i]
            print("\nTrue Value: " + video_name[0:3] + " Prediction: " + ''.join(finalword))

            time.sleep(1)
            pred_array.append([''.join(finalword), gold_label])

        df = DataFrame(pred_array, columns=['pred', 'true'])
        print(classification_report(df.pred, df.true))
        df.to_csv(os.path.join(self.wordVideoPath, 'results.csv'))