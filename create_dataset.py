# Importing Required Libraries:
import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Load different parts of the MediaPipe library.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# Specifies the directory where the image data is stored.
DATA_DIR = './data'


# Empty lists to store the hand landmark data and their corresponding labels (class names).
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []


# Processing Each Image:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Processing the Image to Detect Hand Landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Normalizing the Coordinates
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Storing the Data
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # storing Data and Labels
            data.append(data_aux)
            labels.append(dir_)

# save the dataset, it consisting of the processed data and labels
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
