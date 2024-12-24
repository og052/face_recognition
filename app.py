import cv2



def start():

    url = 0
    cap = cv2.VideoCapture(url)
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

    dirr = {}

    with open("text.txt", "r") as file:
        for line in file:
            line = line.strip()
            key, val = line.split(":")
            key = key[1:]
            val = val[2:-2]
            dirr[int(key)]=val

    while True:
        ret, frame = cap.read()

        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detect.detectMultiScale(gray,1.1,3)


            #cv2.rectangle(frame, (x_value,y_value), (x_value+h_value,y_value+h_value), (255,0,0), 3)

            for(x,y,w,h) in faces:
                serial, conf = recognizer.predict(gray[y:y+w, x:x+h])
                if conf <40:

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"Match!{dirr[serial]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "No Match!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)

        else:
            print("Failed to grab frame")
            break

        k = cv2.waitKey(1)

        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
start()