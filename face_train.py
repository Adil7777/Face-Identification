import os
from PIL import Image
import numpy as np
import cv2
import pickle


class Train:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('cascedes/data/haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainner.yml')
        self.current_id = 0
        self.label_ids = {}
        self.y_labels = []
        self.x_train = []

    def get_path(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, 'images')

        return self.image_dir

    def get_arrays(self):
        image_dir = self.get_path()

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('jpg') or file.endswith('png'):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()

                    print(label, path)

                    if label not in self.label_ids:
                        self.label_ids[label] = self.current_id
                        self.current_id += 1

                    id_ = self.label_ids[label]

                    pil_image = Image.open(path).convert("L")

                    image_array = np.array(pil_image, 'uint8')

                    face = self.face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                    for (x, y, w, h) in face:
                        roi = image_array[y:y + h, x:x + w]
                        self.x_train.append(roi)
                        self.y_labels.append(id_)

        self.save_pickle()

    def save_pickle(self):
        with open('labels.pickle', 'wb') as f:
            pickle.dump(self.label_ids, f)
        self.train()

    def train(self):
        self.recognizer.train(self.x_train, np.array(self.y_labels))
        self.recognizer.save('trainner.yml')


train = Train()
train.get_arrays()

