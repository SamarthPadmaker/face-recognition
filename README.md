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
