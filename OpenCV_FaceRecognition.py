import cv2


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: 
names = ['None', 'Ilya Smolin', 'Smolin']

# Initialize and start realtime video capture
video = cv2.VideoCapture(0)
video.set(3, 640)  # set video widht
video.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * video.get(3)
minH = 0.1 * video.get(4)

while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 255), 1)

    cv2.imshow('Face recognition', img)

    key = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()