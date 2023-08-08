import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 30
imgSize = 300
count = 0

folder = r'D:\Vietname_sig\Data\Hello'

while True:
    ret, frame = cap.read()
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = frame[y - offset: y + h + offset, x - offset: x + w + offset]
        
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:

            imgShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wGap + wCal] = imgResize

            else:
                k = imgSize / h
                hCal = math.ceil(k * w)
                imgResize = cv.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hGap + hCal, :] = imgResize

            cv.imshow('ImageCrop', imgCrop)
            cv.imshow("ImgWhite", imgWhite)

    cv.imshow("Image", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        count += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print('Image saved: ', count)