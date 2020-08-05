import cv2
import pickle
import os


class FaceDetection:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade, self.eye_cascade = self.cascades()
        self.new_labels = self.get_dict()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainner.yml')

    def cascades(self):
        face_cascade = cv2.CascadeClassifier('cascedes/data/haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('cascedes/data/haarcascade_eye.xml')
        return face_cascade, eye_cascade

    def get_dict(self):
        labels = {"person_name": 1}
        if os.path.getsize('labels.pickle') > 0:
            with open('labels.pickle', "rb") as f:
                unpickler = pickle.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                labels = unpickler.load()
        # print(type(labels))

        new_labels = {}
        counter = 0
        for i in list(labels.keys()):
            new_labels[counter] = i
            counter += 1

        return new_labels

    def main(self):
        while 1:
            # Capture frame by frame
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                print(x, y, w, h)
                roi_gray = gray[y:y + h, x:x + h]
                roi_color = frame[y:y + h, x:x + h]
                # if counter < 1000:
                #     img_item = 'images/adil/{}.png'.format(str(counter))
                #     cv2.imwrite(im  M
                #     0g_item, frame)
                #     counter += 1
                # else:
                #     break

                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 45 and conf <= 85:
                    name = self.new_labels[id_]
                else:
                    name = 'Unknown'

                font = cv2.FONT_HERSHEY_COMPLEX
                color = (255, 255, 255)
                stroke = 1

                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                color = (255, 0, 0)
                stroke = 2
                w += x
                h += y
                cv2.rectangle(frame, (x, y), (w, h), color, stroke)
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        self.destroy()

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()


a = FaceDetection()
a.main()
