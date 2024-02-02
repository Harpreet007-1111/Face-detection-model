import cv2
import os

def detect_face_and_gender(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    script_dir = os.path.dirname(os.path.abspath("Happy.jpg"))
    prototxt_path = os.path.join(script_dir, 'deploy_gender.prototxt')
    caffemodel_path = os.path.join(script_dir, 'gender_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]

        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        # Get gender
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'Gender: {gender}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display image
    cv2.imshow('Face and Gender Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_face_and_gender('Happy.jpg')
