import cv2
import face_recognition
import numpy as np
from fer import FER  # Make sure to install this library

class VideoCamera(object):
    def __init__(self, known_face_encodings, known_face_names):
        self.video = cv2.VideoCapture(0)
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.detector = FER()  # Initialize the emotion detector

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                color = (0, 0, 255)  # Red for recognized faces
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, "Match Found", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            else:
                color = (255, 0, 0)  # Blue for unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, "Unknown", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_emotion_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion_data = self.detector.detect_emotions(rgb_frame)

        for face in emotion_data:
            bounding_box = face['box']
            emotions = face['emotions']
            emotion = max(emotions, key=emotions.get)
            score = emotions[emotion]

            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({score:.2f})", (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
