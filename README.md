# Automatic Face Capture and Recognition

This repository contains a Python-based implementation for automatic face capture and recognition using a webcam. It uses DeepFace for face embeddings and MongoDB for storing and retrieving embeddings for recognition. The system performs real-time face capture, generates embeddings, and compares them with pre-stored embeddings in the database.

# Features

- Automatic Face Capture: Captures a face using the system's default webcam.

- Face Embedding Generation: Generates embeddings using DeepFace with the Facenet model.

- Face Recognition: Compares captured embeddings with stored embeddings in MongoDB to find a match.

- MongoDB Integration: Stores and retrieves face embeddings for persistent face data management.

# Prerequisites

- Python 3.8 or above

- MongoDB installed locally or accessible remotely

# Python Libraries

Install the required libraries using pip:
 - ``` pip install opencv-python-headless deepface pymongo numpy ```

#  How It Works

# Face Capture and Recognition Script

1. Automatically captures an image from the webcam.

2. Generates an embedding for the captured image using DeepFace.

3. Compares the embedding with stored embeddings in MongoDB.

4. Outputs the best match if found within the threshold.

# MongoDB Population Script

- Processes images from a specified folder.

- Generates embeddings for each image.

- Saves the embeddings to MongoDB with the corresponding filename.

# Usage

# MongoDB Population

To populate MongoDB with embeddings from images:
1. Update the folder path in the script
2. Run the script to extract embeddings and store them in MongoDB
- ```python mongo_population_script.py```

# Face Capture and Recognition

To perform automatic face capture and recognition:

1. Run the script:
 -  ```python face_recognition_script.py```
2. The script will automatically capture an image and compare it with stored embeddings.

# File Structure

- face_recognition_script.py: Handles face capture, embedding generation, and recognition.

- mongo_population_script.py: Populates MongoDB with face embeddings from a folder.
