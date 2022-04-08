import glob
import sys

import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
import torch


def getVectorForAlphabet(files):

    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    video_names = []
    step = int(len(files) / 100)
    if step == 0:
        step = 1

    count = 0
    for frame in files:
        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = model.extract_feature(img)
        results = np.squeeze(results)
        predicted = np.where(results==max(results))[0][0]

        vectors.append(predicted)
        video_names.append(os.path.basename(frame))
        count += 1
        if count % step == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
    return vectors

def loadLabels(labelFile):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def loadLabelDictionary(labelFile):
    id_to_labels = loadLabels(labelFile)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id


def predictFromFrame(videosPath):
    files = []
    path = os.path.join(videosPath, "*.png")
    frames = glob.glob(path)
    frames.sort()
    files = frames
    prediction_vector = getVectorForAlphabet(files)
    label_file = 'expectedOutputAlphas.txt'
    id_to_labels, labels_to_id = loadLabelDictionary(label_file)
    final_predictions=[]
    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    return final_predictions

def predict_words_from_frames(path, start, end):
    files=[]
    for i in range(start, end + 1, 1):
        try:
            path = os.path.join(path, str(i) + ".png")
            frames = glob.glob(path)
            files.append(frames[0])
        except:
            continue
    alphaFeatureVector = getVectorForAlphabet(files)
    groundTruthsFilePath = 'expectedOutputWords.txt'


    id_to_labels, labels_to_id = loadLabelDictionary(groundTruthsFilePath)

    final_predictions=[]
    for i in range(len(alphaFeatureVector)):
        for ins in labels_to_id:
            if alphaFeatureVector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    return final_predictions
