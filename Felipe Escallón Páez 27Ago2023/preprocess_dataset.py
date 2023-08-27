import os
from tqdm import tqdm
import argparse
# Libraries for the face recognition
import cv2 # 4.5.5.64
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN


# Argparse argument for the path to the dataset folder
parser = argparse.ArgumentParser()
parser.add_argument('--face_dataset_path', type=str, default='face_dataset', help='Path to the dataset folder')
args = parser.parse_args()
face_dataset_path = args.face_dataset_path

# Create a processed dataset folder if it doesn't exist
if not os.path.exists(f'processed_{face_dataset_path}'):
    os.makedirs(f'processed_{face_dataset_path}')

# Create a training folder inside the processed dataset folder if it doesn't exist
if not os.path.exists(f'processed_{face_dataset_path}/training'):
    os.makedirs(f'processed_{face_dataset_path}/training')
    # Create a folder for each person inside the training folder
    for name in os.listdir(f'{face_dataset_path}/training'):
        os.makedirs(f'processed_{face_dataset_path}/training/{name}')

# Create a test folder inside the processed dataset folder if it doesn't exist
if not os.path.exists(f'processed_{face_dataset_path}/test'):
    os.makedirs(f'processed_{face_dataset_path}/test')
    # Create a folder for each person inside the test folder
    for name in os.listdir(f'{face_dataset_path}/test'):
        os.makedirs(f'processed_{face_dataset_path}/test/{name}')

# Create function to crop and save faces
def crop_resize_and_save_faces(image_path, new_image_path):
    # Load an image using OpenCV
    image = cv2.imread(image_path)

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

    # Draw bounding boxes around detected faces
    i=0
    for box in filtered_boxes:
        i=i+1
        x, y, w, h, c = box
        margin_factor = 0.15
        section_of_img = image[int(y-int(margin_factor*(h-y))):int(h+int(margin_factor*(h-y))), int(x-int(margin_factor*(w-x))):int(w+int(margin_factor*(w-x))), :]
        resized_section_of_img = cv2.resize(section_of_img, (400, 500))
        cv2.imwrite(f'{new_image_path}_{i}.jpg', resized_section_of_img)

# Iterate through the images in a folder inside the dataset folder
def process_images(fold):
    for name in tqdm(os.listdir(f'{face_dataset_path}/{fold}')):
        for image in os.listdir(f'{face_dataset_path}/{fold}/{name}'):
            # Crop and save faces
            crop_resize_and_save_faces(f'{face_dataset_path}/{fold}/{name}/{image}', f'processed_{face_dataset_path}/{fold}/{name}/{image}')

# Process images in the training folder
process_images('training')

# Process images in the test folder
process_images('test')

