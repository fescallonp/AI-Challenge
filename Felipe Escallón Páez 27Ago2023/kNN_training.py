import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import argparse
import joblib
import json

# Argparse argument for the path to the dataset folder
parser = argparse.ArgumentParser()
parser.add_argument('--face_dataset_training_path', type=str, default='processed_face_dataset/training', help='Path to the dataset folder')
parser.add_argument('--face_dataset_test_path', type=str, default='processed_face_dataset/test', help='Path to the dataset folder')
args = parser.parse_args()
training_data_dir = args.face_dataset_training_path
test_data_dir = args.face_dataset_test_path

people = os.listdir(training_data_dir)

X_train = []  # List to store face data
y_train = []  # List to store corresponding labels
X_test = []   # List to store face data
y_test = []   # List to store corresponding labels

for person_id, person in enumerate(people):
    training_person_dir = os.path.join(training_data_dir, person)
    for image_file in os.listdir(training_person_dir):
        image_path = os.path.join(training_person_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        resized_image = cv2.resize(image, (400, 500))  # Resize for consistent size
        X_train.append(np.array(resized_image).flatten())  # Flatten image data
        y_train.append(person_id)
    test_person_dir = os.path.join(test_data_dir, person)
    for image_file in os.listdir(test_person_dir):
        image_path = os.path.join(test_person_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (400, 500))
        X_test.append(np.array(resized_image).flatten())
        y_test.append(person_id)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

joblib.dump(knn_classifier, 'face_identification_model.joblib')

for i in range(len(X_test)):
    print("===========================================")
    print(f"Predicted: {people[knn_classifier.predict([X_test[i]])[0]]}")
    print(f"Actual: {people[y_test[i]]}")

index_to_people = {}
for i, person in enumerate(people):
    index_to_people[i] = person
# Save dictionary of the index to people's names as a json file
with open('index_to_people.json', 'w') as f:
    json.dump(index_to_people, f)

print("===========================================")
accuracy = knn_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
print("===========================================")






