import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#testing comments

path = 'image'
image = []
personName = []
myList = os.listdir(path)
print(myList)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    image.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
    print(personName)


def faceEncoddings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodelistknown = faceEncoddings(image)
print("all encoding completes!!")


def attendence(name):

    with open('attendence.csv', 'r+') as f:
        

        myDataList = f.readlines()
        nameList = []
        for line in myDataList:

            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%y')
            f.writelines(f' {name},{tStr},{dStr}')


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facescurrentframe = face_recognition.face_locations(faces)
    encodecurrentframe = face_recognition.face_encodings(
        faces, facescurrentframe)

    for encodeface, faceLoc in zip(encodecurrentframe, facescurrentframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        faceDic = face_recognition.face_distance(encodelistknown, encodeface)

        matchindex = np.argmin(faceDic)

        if matches[matchindex]:
            name = personName[matchindex].upper()
            # print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)
            attendence(name)

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()
