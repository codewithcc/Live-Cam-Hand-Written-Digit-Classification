import time as t
import cv2 as cv
import pickle as pk
from tensorflow import keras as kr
import numpy as np

if __name__ == '__main__':
    timeS = timeE = 0

    model_nn = kr.models.load_model(r'model_nn.h5')

    cam = cv.VideoCapture(0, 700)

    while cam.isOpened():
        _, image = cam.read()

        if not _: continue

        height, width, channel = image.shape
        bbox = (width // 2) - 30, (height // 2) - 30, (width // 2) + 60, (height // 2) + 60
        cv.rectangle(image, (bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5), (0, 255, 0), 2)
        image_crop = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        image_crop = cv.rotate(image_crop, cv.ROTATE_180)
        image_crop = cv.cvtColor(image_crop, cv.COLOR_BGR2GRAY)
        thrVal, image_crop = cv.threshold(image_crop, 120, 255, cv.THRESH_BINARY_INV)

        image_show = cv.resize(image_crop, (200, 200))
        image_test_784 = cv.resize(image_crop, (28, 28)).flatten()
        image_test_784 = image_test_784 / 255
        # print(image_test_64.shape)
        # print(image_test_784)

        try:
            pred = str(np.argmax(model_nn.predict(image_test_784.reshape(1, 784))[0]))
            accr = str(round(np.amax(model_nn.predict(image_test_784.reshape(1, 784))) * 100, 2))

            cv.rectangle(image, (width - 200, 200), (width, height), (0, 255, 0), -1)
            cv.putText(image, pred, (width - 160, 380), 1, 10, (255, 0, 0), 5)
            cv.putText(image, f'{accr} %', (width - 170, 450), 1, 2.6, (255, 0, 0), 3)
        except Exception as e:
            print(f'Error : {e}')

        image[0: 200, width - 200: width] = cv.cvtColor(image_show, cv.COLOR_GRAY2BGR)

        timeE = t.time()
        fps = 1 // (timeE - timeS)
        timeS = timeE
        cv.putText(image, str(f'FPS : {fps}'), (10, 40), 2, 1, (255, 0, 255), 2)

        cv.imshow('Original Image', image)

        if cv.waitKey(1) & 0xff == ord('q'): break

    cam.release()
    cv.destroyAllWindows()