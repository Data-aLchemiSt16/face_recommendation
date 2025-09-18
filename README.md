# face_recommendation

import cv2
import os
import numpy as np
from PIL import Image

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_classifier.empty():
        print("Error loading haarcascade_frontalface_default.xml")
        return

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    if not os.path.exists("facedata"):
        os.makedirs("facedata")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # fix the id = 1 in generate_dataset function (each time generate dataset is used, if it's a new student, a new id should be generated)
    existing_ids = [int(f.split(".")[1]) for f in os.listdir("facedata") if f.startswith("user.")]
    id = max(existing_ids, default=0) + 1   # New student gets next ID

    img_id = 0
    print(f"Starting face capture for ID: {id}. Press Enter or wait for 200 samples.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = f"facedata/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)

        if cv2.waitKey(1) == 13 or img_id == 200:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Samples have been collected for ID: {id}")

if __name__ == "__main__":
    generate_dataset()

def Train_Classifier(facedata_dir):
    path = [os.path.join(facedata_dir,f) for f in os.listdir(facedata_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        ImageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(ImageNp)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")

Train_Classifier("facedata")

def DrawBoundary(img, classifier, scaleFactor, minNeighbor, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbor)
    coords = []

    for (x, y, w, h) in features:
        id, pred = clf.predict(gray_image[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        # Green rectangle if known, else Red for unknown 
        if confidence > 75:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            cv2.putText(img, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  
            cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        coords = [x, y, w, h]

    return coords

def recognize(img, clf, face_cascade):
    coords = DrawBoundary(img, face_cascade, 1.2, 10, clf)
    return img

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error loading haarcascade_frontalface_default.xml")
    exit()

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    img = recognize(img, clf, face_cascade)
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) == 13:  
        break

video_capture.release()
cv2.destroyAllWindows()
