import json

import cv2
import numpy as np
import pandas as pd
import os

SCORE_OVERALL = "score_overall",
NOSE_SCORE = "nose_score",
NOSE_X = "nose_x",
NOSE_Y = "nose_y",
LEFTEYE_SCORE = "leftEye_score",
LEFTEYE_X = "leftEye_x",
LEFTEYE_Y = "leftEye_y",
RIGHTEYE_SCORE = "rightEye_score",
RIGHTEYE_X = "rightEye_x",
RIGHTEYE_Y = "rightEye_y",
LEFTEAR_SCORE = "leftEar_score",
LEFTEAR_X = "leftEar_x",
LEFTEAR_Y = "leftEar_y",
RIGHTEAR_SCORE = "rightEar_score",
RIGHTEAR_X = "rightEar_x",
RIGHTEAR_Y = "rightEar_y",
LEFTSHOULDER_SCORE = "leftShoulder_score",
LEFTSHOULDER_X = "leftShoulder_x",
LEFTSHOULDER_Y = "leftShoulder_y",
RIGHTSHOULDER_SCORE = "rightShoulder_score",
RIGHTSHOULDER_X = "rightShoulder_x",
RIGHTSHOULDER_Y = "rightShoulder_y",
LEFTELBOW_SCORE = "leftElbow_score",
LEFTELBOW_X = "leftElbow_x",
LEFTELBOW_Y = "leftElbow_y",
RIGHTELBOW_SCORE = "rightElbow_score",
RIGHTELBOW_X = "rightElbow_x",
RIGHTELBOW_Y = "rightElbow_y",
LEFTWRIST_SCORE = "leftWrist_score",
LEFTWRIST_X = "leftWrist_x",
LEFTWRIST_Y = "leftWrist_y",
RIGHTWRIST_SCORE = "rightWrist_score",
RIGHTWRIST_X = "rightWrist_x",
RIGHTWRIST_Y = "rightWrist_y",
LEFTHIP_SCORE = "leftHip_score",
LEFTHIP_X = "leftHip_x",
LEFTHIP_Y = "leftHip_y",
RIGHTHIP_SCORE = "rightHip_score",
RIGHTHIP_X = "rightHip_x",
RIGHTHIP_Y = "rightHip_y",
LEFTKNEE_SCORE = "leftKnee_score",
LEFTKNEE_X = "leftKnee_x",
LEFTKNEE_Y = "leftKnee_y",
RIGHTKNEE_SCORE = "rightKnee_score",
RIGHTKNEE_X = "rightKnee_x",
RIGHTKNEE_Y = "rightKnee_y",
LEFTANKLE_SCORE = "leftAnkle_score",
LEFTANKLE_X = "leftAnkle_x",
LEFTANKLE_Y = "leftAnkle_y",
RIGHTANKLE_SCORE = "rightAnkle_score",
RIGHTANKLE_X = "rightAnkle_x",
RIGHTANKLE_Y = "rightAnkle_y"
SCORE = "score"
KEYPOINTS = "keypoints"
POSITION = "position"
X = "x"
Y = "y"

def getCsvFromJson(dirPath):
    cols = [SCORE_OVERALL,
               NOSE_SCORE,
               NOSE_X,
               NOSE_Y,
               LEFTEYE_SCORE,
               LEFTEYE_X,
               LEFTEYE_Y,
               RIGHTEYE_SCORE,
               RIGHTEYE_X,
               RIGHTEYE_Y,
               LEFTEAR_SCORE,
               LEFTEAR_X,
               LEFTEAR_Y,
               RIGHTEAR_SCORE,
               RIGHTEAR_X,
               RIGHTEAR_Y,
               LEFTSHOULDER_SCORE,
               LEFTSHOULDER_X,
               LEFTSHOULDER_Y,
               RIGHTSHOULDER_SCORE,
               RIGHTSHOULDER_X,
               RIGHTSHOULDER_Y,
               LEFTELBOW_SCORE,
               LEFTELBOW_X,
               LEFTELBOW_Y,
               RIGHTELBOW_SCORE,
               RIGHTELBOW_X,
               RIGHTELBOW_Y,
               LEFTWRIST_SCORE,
               LEFTWRIST_X,
               LEFTWRIST_Y,
               RIGHTWRIST_SCORE,
               RIGHTWRIST_X,
               RIGHTWRIST_Y,
               LEFTHIP_SCORE,
               LEFTHIP_X,
               LEFTHIP_Y,
               RIGHTHIP_SCORE,
               RIGHTHIP_X,
               RIGHTHIP_Y,
               LEFTKNEE_SCORE,
               LEFTKNEE_X,
               LEFTKNEE_Y,
               RIGHTKNEE_SCORE,
               RIGHTKNEE_X,
               RIGHTKNEE_Y,
               LEFTANKLE_SCORE,
               LEFTANKLE_X,
               LEFTANKLE_Y,
               RIGHTANKLE_SCORE,
               RIGHTANKLE_X,
               RIGHTANKLE_Y]
    data = json.loads(open(os.path.join(dirPath, 'key_points.json'), 'r').read())
    csv_data = np.zeros((len(data), len(cols)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i][SCORE])
        for obj in data[i][KEYPOINTS]:
            one.append(obj[SCORE])
            one.append(obj[POSITION][X])
            one.append(obj[POSITION][Y])
        csv_data[i] = np.array(one)
    pd.DataFrame(csv_data, columns=cols).to_csv(os.path.join(dirPath, 'key_points.csv'), index_label='Frames#')

def getFrames(dirPath, framesDirPath):
    video_files = os.listdir(dirPath)
    dictCount = {}
    for file in video_files:
        try:
            if os.path.splitext(file)[1] !='.mp4':
                continue
            print('Extracting Frame for file: {}'.format(file))
            video = cv2.VideoCapture(os.path.join(dirPath, file))
            label = getAlphabetLabelFromFile(file)
            count = dictCount.get(label,0)
            success = 1
            arr_img = []
            if not os.path.isdir(os.path.join(framesDirPath, label)):
                os.mkdir(os.path.join(framesDirPath, label))
            new_path = os.path.join(framesDirPath, label)
            while success:
                success, image = video.read()
                arr_img.append(image)
                count += 1
            for i in range(len(arr_img)-1):
                image_path = os.path.join(new_path,"%d.png" % count)
                cv2.imwrite(image_path, arr_img[i])
                count += 1
            dictCount[label] = count+1
        except:
            continue


def getAlphabetLabelFromFile(filename):
    return os.path.split(filename)[1].replace(".mp4","").upper()