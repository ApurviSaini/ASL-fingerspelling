import os
from AlphabetPredictionUtils import predictFromFrame
from AlphabetPredictionUtils import predictWordsFromFrames
from statistics import mode
from pandas import DataFrame
from sklearn.metrics import classification_report

class AlphabetPrediction:

    def __init__(self,alphabetVideoPath='',alphabetFramePath='') :
        self.alphabetVideoPath = alphabetVideoPath
        self.alphabetFramePath = alphabetFramePath

    def predict(self):
        videos = os.listdir(self.alphabetVideoPath)
        if not os.path.exists(self.alphabetFramePath):
            os.makedirs(self.alphabetFramePath)
        groundTruths = []
        for video_name in videos:
            if video_name == '.DS_Store':
                continue
            print("Running Predictions for " + video_name)
            file_path = os.path.join(self.alphabetVideoPath, video_name)

            test_data = os.path.join(self.alphabetFramePath, video_name.split('.')[0] + "_cropped")
            pred = predictFromFrame(test_data)
            try:
                prediction = mode(pred)
            except:
                prediction = ''
            gold_label = video_name[0]
            print("\nTrue Value: " + video_name[0] + " Prediction: " + prediction)
            groundTruths.append([prediction, gold_label])

        df = DataFrame(groundTruths, columns=['pred', 'true'])
        print(classification_report(df.pred, df.true))
        df.to_csv(os.path.join(self.alphabetVideoPath, 'results.csv'))