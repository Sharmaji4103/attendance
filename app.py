import os
import cv2
import face_recognition as fr
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join('static', 'faculty_images')
ALLOWED_EXTENSIONS = {'jpg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to control facial recognition state
facial_recognition_running = False
cap = None  # Webcam capture variable
lock = threading.Lock()  # To ensure thread safety

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = fr.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.txt', 'a') as f:
        now = datetime.now()
        date_time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name}, {date_time_string}\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_us():
    return render_template('about_us.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                flash('File with this name already exists. Please choose a different name.')
                return redirect(request.url)
            file.save(file_path)
            flash('File successfully uploaded!')
            return redirect(url_for('index'))
        else:
            flash('Only .jpg files are allowed!')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/run_recognition', methods=['POST'])
def run_face_recognition():
    global facial_recognition_running, cap
    if facial_recognition_running:
        flash("Facial recognition is already running.")
        return redirect(url_for('index'))

    with lock:
        facial_recognition_running = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        flash("Error: Could not open webcam.")
        return redirect(url_for('index'))

    # Process the facial recognition in a separate thread
    def process_recognition():
        global facial_recognition_running
        path = app.config['UPLOAD_FOLDER']
        images = []
        Names = []
        myList = os.listdir(path)

        for cl in myList:
            curImg = cv2.imread(os.path.join(path, cl))
            if curImg is not None:
                images.append(curImg)
                Names.append(os.path.splitext(cl)[0])

        encoded_face_train = findEncodings(images)
        recognized_names = set()

        try:
            while facial_recognition_running:
                success, img = cap.read()
                if not success:
                    flash("Failed to capture image.")
                    break

                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = fr.face_locations(imgS)
                encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = fr.compare_faces(encoded_face_train, encodeFace)
                    faceDis = fr.face_distance(encoded_face_train, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = Names[matchIndex].upper()
                        if name not in recognized_names:
                            markAttendance(name)
                            recognized_names.add(name)

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Webcam', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            with lock:
                facial_recognition_running = False

    recognition_thread = threading.Thread(target=process_recognition)
    recognition_thread.start()

    return redirect(url_for('index'))

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global facial_recognition_running
    if not facial_recognition_running:
        flash("Facial recognition is not running.")
    else:
        with lock:
            facial_recognition_running = False
        flash("Facial recognition stopped.")

    return redirect(url_for('index'))

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    attendance_records = []
    try:
        with open('Attendance.txt', 'r') as f:
            attendance_records = f.readlines()
    except FileNotFoundError:
        attendance_records = []

    return render_template('view_attendance.html', records=attendance_records)

@app.route('/manage_attendance')
def manage_attendance():
    attendance_records = []
    try:
        with open('Attendance.txt', 'r') as f:
            attendance_records = f.readlines()
    except FileNotFoundError:
        attendance_records = []

    return render_template('manage_attendance.html', records=attendance_records)

@app.route('/delete_attendance/<int:record_id>', methods=['POST'])
def delete_attendance(record_id):
    attendance_records = []
    try:
        with open('Attendance.txt', 'r') as f:
            attendance_records = f.readlines()
    except FileNotFoundError:
        flash('Attendance file not found.')
        return redirect(url_for('manage_attendance'))

    if record_id < 0 or record_id >= len(attendance_records):
        flash('Invalid record ID.')
        return redirect(url_for('manage_attendance'))

    # Remove the record from the list
    attendance_records.pop(record_id)

    # Write the updated records back to the file
    with open('Attendance.txt', 'w') as f:
        f.writelines(attendance_records)

    flash('Attendance record deleted successfully.')
    return redirect(url_for('manage_attendance'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
