# FaceRecognition
Overview

This project is a face recognition system that uses DeepFace for facial verification and OpenCV for video capture and display. It continuously scans faces from a webcam feed and compares them to reference images stored locally, identifying users in real-time.

Features:
-Real-time face recognition
-Multi-threaded face verification
-Automatic downloading of required face recognition models if not already present
-Adjustable reference image database

Requirements;
  Libraries:
    -cv2 (OpenCV): Handles video capture and image processing.
    -deepface: Performs face recognition tasks.
    -threading: Manages simultaneous processes.
    -os: Handles system operations.

Installation:
pip install opencv-python deepface

How to Use:
  Set Up Reference Images:
    -Add reference images in the designated paths within the dic_ref dictionary in the code.

How It Works:
  Model Preparation:
    Checks if required DeepFace models (VGG-Face, Facenet, OpenFace) are downloaded.
    Automatically downloads missing models.

  Face Verification Process:
    -Captures video frames from the webcam.
    -Every 30 frames, starts a verification thread.
    -Compares captured frames against reference images.
    -Displays match results in real-time.
  Error Handling:
    -Catches and logs any face verification errors.
