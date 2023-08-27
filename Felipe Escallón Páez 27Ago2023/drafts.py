# Libraries for the face recognition
import cv2 # 4.5.5.64
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

# fer+ pre-trained model for emotion recognition on the detected face
# emotions: neutral, happiness, surprise, sadness, anger, disgust, fear.
from fer_pytorch.fer import FER # https://github.com/Emilien-mipt/fer-pytorch

# Load a pre-trained MTCNN model
mtcnn = MTCNN()

# Load an image using OpenCV
image_path = 'prueba.JPG'
image = cv2.imread(image_path)

# Convert image from BGR to RGB (required for MTCNN)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection
boxes, scores = mtcnn.detect(image)

# Put boxes and scores in a same list for each face
boxes = [[x, y, x1, y1, score] for x, y, x1, y1, score in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)]

# Set a confidence threshold for face detection
confidence_threshold = 0 # Adjust as needed

# Filter detections based on confidence threshold
filtered_boxes = [box for box in boxes if box[4] >= confidence_threshold]

# Draw bounding boxes around detected faces
i=0
faces_list = []
coordinates_per_faces_list = {}
for box in filtered_boxes:
    i=i+1
    x, y, w, h, c = box
    margin_factor = 0.15
    cv2.imwrite(f'detected_face_{i}.jpg', image[int(y-int(margin_factor*(h-y))):int(h+int(margin_factor*(h-y))), int(x-int(margin_factor*(w-x))):int(w+int(margin_factor*(w-x))), :])
    faces_list.append(f'detected_face_{i}.jpg')
    coordinates_per_faces_list[f'detected_face_{i}.jpg'] = [int(x-int(margin_factor*(w-x))), int(y-int(margin_factor*(h-y))), int(w+int(margin_factor*(w-x))), int(h+int(margin_factor*(h-y)))]
    cv2.rectangle(image, (int(x-int(margin_factor*(w-x))), int(y-int(margin_factor*(h-y)))), (int(w+int(margin_factor*(w-x))), int(h+int(margin_factor*(h-y)))), (0, 0, 255), 3)

# ------------------- EMOTION RECOGNITION -------------------
fer = FER()
fer.get_pretrained_model("resnet34")

for face in faces_list:
    img = cv2.imread(face)
    result = fer.predict_image(img)
    # Choose the key with the highest value from result['emotions']
    # Define font size and width to fit the number of pixels of the width of the face
    text_size = 0.008*(coordinates_per_faces_list[face][2]-coordinates_per_faces_list[face][0])
    text_width = int(0.008*(coordinates_per_faces_list[face][2]-coordinates_per_faces_list[face][0]))
    try:
        # print(result)
        # print(face, ': ', max(result['emotions'], key=result['emotions'].get))
        # Write over the image the emotion detected
        cv2.putText(image, max(result['emotions'], key=result['emotions'].get), (coordinates_per_faces_list[face][0], coordinates_per_faces_list[face][1]-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_width)
    except:
        # print(result[0])
        # print(face, ': ', max(result[0]['emotions'], key=result[0]['emotions'].get))
        # Write over the image the emotion detected
        cv2.putText(image, max(result[0]['emotions'], key=result[0]['emotions'].get), (coordinates_per_faces_list[face][0], coordinates_per_faces_list[face][1]-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_width)

cv2.imwrite(f'result_{image_path}', image)

# # Convert image back to RGB for displaying with matplotlib
# image_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Display the image with bounding boxes
# plt.imshow(image_with_boxes)
# plt.axis('off')
