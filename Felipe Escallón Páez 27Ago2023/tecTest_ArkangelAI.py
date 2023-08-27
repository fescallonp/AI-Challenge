import os
import joblib
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# Libraries for the face recognition
import cv2 # 4.5.5.64
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

# fer+ pre-trained model for emotion recognition on the detected face
# emotions: neutral, happiness, surprise, sadness, anger, disgust, fear.
from fer_pytorch.fer import FER # https://github.com/Emilien-mipt/fer-pytorch

# Argparse for the command line arguments such as model_filename 
parser = argparse.ArgumentParser(description='Face recognition and emotion recognition')
parser.add_argument('--model_filename', type=str, default='face_identification_model.joblib', help='Name of the model file')
parser.add_argument('--image', type=str, default='candidatos.png', help='Name of the image file')
argparse = parser.parse_args()

class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

people = json.load(open('index_to_people.json'))

# Example new image
image = cv2.imread(argparse.image)

# Load a pre-trained MTCNN model
mtcnn = MTCNN()

# Perform face detection
boxes, scores = mtcnn.detect(image)

# Put boxes and scores in a same list for each face
boxes = [[x, y, x1, y1, score] for x, y, x1, y1, score in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)]

# Set a confidence threshold for face detection
confidence_threshold = 0.9 # Adjust as needed

# Filter detections based on confidence threshold
filtered_boxes = [box for box in boxes if box[4] >= confidence_threshold]

coordinates_per_faces_list = {}
i=0
# Analyze each detected face
for box in filtered_boxes:
    i=i+1
    x, y, w, h, c = box
    margin_factor = 0.15
    face = f'detected_face_{i}.jpg'
    coordinates_per_faces_list[face] = [int(x-int(margin_factor*(w-x))), int(y-int(margin_factor*(h-y))), int(w+int(margin_factor*(w-x))), int(h+int(margin_factor*(h-y)))]
    cv2.rectangle(image, (int(x-int(margin_factor*(w-x))), int(y-int(margin_factor*(h-y)))), (int(w+int(margin_factor*(w-x))), int(h+int(margin_factor*(h-y)))), (0, 0, 255), 3)
    actual_face = image[int(y-int(margin_factor*(h-y))):int(h+int(margin_factor*(h-y))), int(x-int(margin_factor*(w-x))):int(w+int(margin_factor*(w-x))), :]
    if argparse.model_filename == 'face_identification_model.pth':
        num_classes = len(people)
        model = FaceCNN(num_classes)
        model.load_state_dict(torch.load('face_identification_model.pth'))
        model.eval()  # Set the model to evaluation mode

        resized_image = cv2.resize(actual_face, (100, 100)).transpose(2, 0, 1)
        input_batch = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)
        predicted_label = torch.argmax(output).item()
        predicted_person = people[f'{predicted_label}']
    elif argparse.model_filename == 'face_identification_model.joblib':
        loaded_model = joblib.load(argparse.model_filename)
        actual_face = cv2.cvtColor(actual_face, cv2.COLOR_BGR2GRAY)
        resized_actual_face = cv2.resize(actual_face, (400, 500))
        flattened_actual_face = np.array(resized_actual_face).flatten()
        predicted_label = loaded_model.predict([flattened_actual_face])
        predicted_person = people[f'{predicted_label[0]}']
    # Write on the image the name of the person detected
    text_size = 0.003*(coordinates_per_faces_list[face][2]-coordinates_per_faces_list[face][0])
    text_width = int(0.008*(coordinates_per_faces_list[face][2]-coordinates_per_faces_list[face][0]))
    cv2.putText(image, predicted_person, (int(x-int(margin_factor*(w-x))), int(y-int(margin_factor*(h-y)))), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_width)
    # ------------------- EMOTION RECOGNITION -------------------
    fer = FER()
    fer.get_pretrained_model("resnet34")

    gray_actual_face = cv2.cvtColor(actual_face, cv2.COLOR_BGR2RGB)
    result = fer.predict_image(gray_actual_face)
    # Write over the image the emotion detected
    try:
        cv2.putText(image, max(result['emotions'], key=result['emotions'].get), (coordinates_per_faces_list[face][0]+5, coordinates_per_faces_list[face][3]-5), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_width)
    except:
        cv2.putText(image, max(result[0]['emotions'], key=result[0]['emotions'].get), (coordinates_per_faces_list[face][0]+5, coordinates_per_faces_list[face][3]-5), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_width)



cv2.imwrite(f'result_{argparse.image}', image)