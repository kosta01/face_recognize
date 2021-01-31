import cv2
import face_recognition as fr
import numpy as np
from PIL import Image


video_capture = cv2.VideoCapture(0)

kostandina_image = fr.load_image_file( "faces\kostandina.jpg")
kostandina_face_encoding = fr.face_encodings(kostandina_image)[0]
known_face_encondings = [kostandina_face_encoding]
known_face_names = ["kostandina"]
kosta_image = fr.load_image_file("faces\kosta.jpg")
kosta_face_encoding = fr.face_encodings(kosta_image)[0]
known_face_encondings = [kosta_face_encoding]
known_face_names = ["kosta"]
ko_image = fr.load_image_file("faces\ko.jpg")
ko_face_encoding = fr.face_encodings(ko_image)[0]
known_face_encondings = [ko_face_encoding]
known_face_names = ["ko"]


while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "finding name"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
    else:
        if cv2.waitKey() & 0xFF ==ord('n'):
            print( "eshte prezent")
                        
video_capture.release()
cv2.destroyAllWindows()
