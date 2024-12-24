import cv2
import os

url = 0
path = "data/"
id = input("Enter name: ")
id2 = len(os.listdir(path))
counter = 0
dir  = {}
dir[id2] = str(id)
cap = cv2.VideoCapture(url)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

with open("text.txt", "a") as file:
    file.write(str(dir)+"\n")

while True:
    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray,1.1,3)

        for(x,y,w,h) in faces:

            counter = counter + 1
            os.makedirs(os.path.join(path,id), exist_ok=True)
            cv2.imwrite(f"data/{id}/User."+str(id2)+"."+str(counter)+".jpg", gray[y:y+w, x:x+h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imshow("Frame", frame)
    else:
        print("Failed to grab frame")
        break

    k = cv2.waitKey(1)
    if counter >100:
        break

cap.release()
cv2.destroyAllWindows()