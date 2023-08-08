import cv2 as cv
import numpy as np
from keras.models import load_model

model_path = r"Model.h5"

model = load_model(model_path)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera IP.")
    cap.release()
    cv.destroyAllWindows()
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc frame từ camera IP.")
        break
    imgSize = 300
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgResize = cv.resize(imgGray, (imgSize, imgSize))
    imgNormalize = imgResize / 255.0
    imgReshape = imgNormalize.reshape(-1, imgSize, imgSize, 1)

    prediction = model.predict(imgReshape)
    predicted_class = np.argmax(prediction, axis=1)

    font = cv.FONT_HERSHEY_SIMPLEX
    label = ['Hello', 'Thanks', 'ByeBye', 'SeeYou', 'Sorry']
    cv.putText(frame, label[predicted_class[0]], (10, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Camera IP", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
