import cv2
import time

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

id = int(input('Input users ID '))
counter = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x , y), (x + w , y + h), (0,0,255), 2)
        crop_frame = gray[y:y + h, x:x + w]
        counter += 1
        path = 'img/User.' + str(id) + '.' + str(counter) + '.jpg'
        cv2.imwrite(path, crop_frame)
        cv2.imshow('Training', frame)
        time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif counter == 50:
        break
video.release()
cv2.destroyAllWindows()