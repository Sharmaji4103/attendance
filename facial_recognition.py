import os
import cv2
import face_recognition as fr
import numpy as np
from datetime import datetime

# Set the path for the images
path = r'D:\web_pro\facial_attendance_project (1)\facial_attendance_project (1)\faculty_images'

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = fr.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

def markAttendance(name):
    try:
        # Append attendance data to the text file
        with open('Attendance.txt', 'a') as f:
            now = datetime.now()
            date_time_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name}, {date_time_string}\n')
    except Exception as e:
        print(f"Error writing to attendance file: {e}")

def run_face_recognition():
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
    
    images = []
    Names = []
    myList = os.listdir(path)

    # Load and encode images
    for cl in myList:
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            Names.append(os.path.splitext(cl)[0])

    encoded_face_train = findEncodings(images)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recognized_names = set()  # Set to track names for marking attendance

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
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
                    if name not in recognized_names:  # Check if name has already been recognized
                        print(f'Attendance marked for: {name}')
                        markAttendance(name)
                        recognized_names.add(name)  # Add name to the set to prevent duplicate marking

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the program.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Call the function to run face recognition
if __name__ == "__main__":
    run_face_recognition()
