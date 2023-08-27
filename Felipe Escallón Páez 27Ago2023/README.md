# Arkangel AI Technical Test

## Requirements

Three main tasks:

- Detection of faces on images.

- Identification of the persons to which the face corresponds.

- Identification of the emotion the person might me experimenting.


## Detection of faces on images

- Taking advantage of pre-trained models over large datasets which have been proven to have high performance, I implemented the facenet_pytorch MTCNN  model’s pre-trained models.

- Additionally, a score threshold of 0.9 is used in order to discard most of the false positives that could be present.


## Identification of the persons to which the face corresponds.

### This task is the one which presents most of the limitations:
- We don’t posses a dataset.
- We don’t know how many people are desired to recognize.
- Training a deep learning model from scratch, depending on the size of the dataset, the resolution of the images, the number of parameters of the network, and the available GPU and CPU resources, could take a lot of epochs and a lot of time. Which currently was very limited.
- **One of the best methods to apply would be to use a model with pre-trained weights over a large dataset such as MSCeleb-1A or VGGFace2 and fine-tune it over the desired dataset.**

### Implementation

- There are three main different types of technologies that could be used: traditional methods such as k-nearest neighbors, CNNs, and transformers. 
- To identify different faces and just for illustrative purposes, I built a small dataset of five of the candidates for mayor of Bogotá with eight examples of each. 
- A script (_preprocess_dataset.py_)  was made to save just the faces instead of the whole images.
- Additionally, the two first techniques were implemented and trained (kNN and a simple CNN) from scratch.

## Identification of the emotion the person might be experimenting.

Just as for the detection of faces, a model with pre-trained weights is used in order to exploit the learnt parameters over large datasets:

- It is the fer+ model taken from https://github.com/Emilien-mipt/fer-pytorch .

- The detected emotions are: neutral, happiness, surprise, sadness, anger, disgust, and fear.


## Results

In red boxes are the misidentified faces and in green are the ones that were adequately identified. Notice that in the building there was no prediction because a face was not predicted and that in the case of the third person from the first row and the fourth person of the second row, they are marked as misidentified but that is the expected result taking into account they were not part of the dataset.


### With the 5 Nearest Neighbors model

![Imagen 1](https://github.com/fescallonp/ArkangelAITest/assets/69943932/0a5c3276-53f9-400c-b121-13da7e976691)


### With the CNN model trained from scratch

![Imagen 2](https://github.com/fescallonp/ArkangelAITest/assets/69943932/1e10230b-68a7-40ad-92db-63a305422bb7)

## Programatically deployable

- Using an AWS EC2 instance I developed the project and in the same way, it can be deployed to obtain the predictions from anywhere through ssh connection.

- You can do so by connecting by terminal:

> ssh -i "arkangelAI_test.pem" ec2-user@ec2-18-117-70-122.us-east-2.compute.amazonaws.com

- To test it you can just run the _tecTest_ArkangelAI.py_ script which by default will load the CNN pre-trained model (_face_identification_model.pth_) and run it over the _candidatos.png_ image which is not part of the dataset and has the faces of all five candidates. The result is an image named _result_candidatos.png_.



##
Developed by Felipe Escallón Páez on August 27 / 2023
