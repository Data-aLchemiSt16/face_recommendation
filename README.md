Face Recognition Project

Features
- Collects face dataset with **auto-incremented IDs**
- Trains a LBPH Classifier
- Detects faces in real-time
- Shows Green rectangle for known faces (ID shown)
- Shows Red rectangle for unknown faces

Usage
1. Run `dataset.py` → Capture face data
2. Run `train.py` → Train classifier
3. Run `recognize.py` → Start recognition

Folder Structure
- `facedata/` → Stores captured face images
- `classifier.xml` → Trained model
- `haarcascade_frontalface_default.xml` → Haarcascade file
