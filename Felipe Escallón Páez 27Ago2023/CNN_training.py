import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
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
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (100, 100))  # Resize for consistent size
        X_train.append(resized_image.transpose(2, 0, 1))  # Transpose to (channels, height, width)
        y_train.append(person_id)
    test_person_dir = os.path.join(test_data_dir, person)
    for image_file in os.listdir(test_person_dir):
        image_path = os.path.join(test_person_dir, image_file)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (100, 100))  # Resize for consistent size
        X_test.append(resized_image.transpose(2, 0, 1))  # Transpose to (channels, height, width)
        y_test.append(person_id)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

train_dataset = FaceDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

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

num_classes = len(people)
model = FaceCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(20)):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'face_identification_model.pth')


model = FaceCNN(num_classes)
model.load_state_dict(torch.load('face_identification_model.pth'))
model.eval()  # Set the model to evaluation mode

test_dataset = FaceDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs, axis=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")

index_to_people = {}
for i, person in enumerate(people):
    index_to_people[i] = person
# Save dictionary of the index to people's names as a json file
with open('index_to_people.json', 'w') as f:
    json.dump(index_to_people, f)
