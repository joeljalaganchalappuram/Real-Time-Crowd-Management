from flask import Flask, render_template, request, redirect, url_for, flash, Response
import os
import face_recognition
import cv2
import numpy as np
import time
from camera import VideoCamera

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

known_face_encodings = []
known_face_names = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = str(time.time()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append("Match Found")
            return redirect(url_for('video_feed'))
        else:
            flash('No face detected in the uploaded image')
            return redirect(request.url)

@app.route('/video_feed')
def video_feed():
    return render_template('video_feed.html')

def generate_frames():
    camera = VideoCamera(known_face_encodings, known_face_names)
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_stream')
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# New route for emotion detection
@app.route('/emotion_detection')
def emotion_detection():
    return render_template('emotion_feed.html')

def generate_emotion_frames():
    camera = VideoCamera(known_face_encodings, known_face_names)
    while True:
        frame = camera.get_emotion_frame()  # This should be defined in camera.py
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/emotion_stream')
def emotion_stream():
    return Response(generate_emotion_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
