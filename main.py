"""
Project : Live Cam Hand-Written Digit Classification
Developer : Chanchal Roy
Date : 1st Jan 2023
GitHub : https://github.com/codewithcc/Live-Cam-Hand-Written-Digit-Classification
"""

# ========== Modules Needed ==========
import time as t
import cv2
from tensorflow import keras as kr
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# ========== Variables ==========
RED = 0, 0, 255
YELLOW = 0, 255, 255
GREEN = 0, 255, 0
BLUE = 255, 0, 0
x, y = 180, 230
pad = 100
timeS = 0
timeE = 0
train = False
myFile = r'ModelCNN.h5'

# ========== Functions ==========
def start_cam(cam_id: int = 0, cam_width: int = 640, cam_height: int = 360) -> object:
    """
    Starts camera.

    cam_id -> Camera id, default 0.
    cam_width -> Width of the pop-up screen, default 640,
    cam_height -> Height of the pop-up screen, default 360,

    It returns the camera object.
    """
    cam = cv2.VideoCapture(cam_id, 700)
    cam.set(3, cam_width)
    cam.set(4, cam_height)
    cam.set(5, 30)
    cam.set(6, cv2.VideoWriter_fourcc(*'MJPG'))
    return cam

def start_training(file_path: str) -> bool:
    """
    Starts the training of the CNN model.

    file_path -> File of the trained model will save.

    Returns True else False if any exception occurs.
    """
    try:
        # ========== Data Loading ==========
        print('Starting the Training...')

        (trainX, trainY), (testX, testY) = kr.datasets.mnist.load_data() # Loads the dataset (keras MNIST has 28X28 image data)

        # ========== Data Pre-processing ==========
        # Normalize the values between 0 and 1
        trainX = trainX / 255
        testX = testX / 255

        # Reshaping the data into (1, 28, 28, 1)
        trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
        testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)

        # Creating a Data Generator
        dataGen = kr.preprocessing.image.ImageDataGenerator(
            width_shift_range=.1,
            height_shift_range=.1,
            zoom_range=.2,
            shear_range=.1,
            rotation_range=10
        )
        dataGen.fit(trainX)
        trainY = kr.utils.to_categorical(trainY, 10)

        # ========== Model Creation ==========
        # Creating the CNN model
        model = kr.Sequential([
            kr.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3), activation='relu'),
            kr.layers.MaxPooling2D((2, 2)),
            kr.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            kr.layers.MaxPooling2D((2, 2)),
            kr.layers.Flatten(),
            kr.layers.Dense(100, activation='relu'),
            kr.layers.Dense(10, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary() # Gets the architecture of the CNN

        model.fit(dataGen.flow(trainX, trainY), epochs=10) # Starts training

        # ========== Model Saving / Loading ==========
        # Saving the model
        try:
            model.save(file_path)
            print('\nModel saved successfully!\n')
        except:
            print('\nError occurred during saving the model!\n')
            return False

        # ========== Model Testing ==========
        # Loading the model
        try:
            model = kr.models.load_model(file_path)
            print('\nModel loaded successfully!\n')
        except:
            print('\nError occurred during loading the model!\n')
            return False

        # One hot encoding
        testY = LabelBinarizer().fit_transform(testY)

        # Checking model performence
        modelResult = model.evaluate(testX, testY, verbose=0)
        print(f'\nModel Loss : {modelResult[0]} | Model Accuracy : {modelResult[1]}\n')

        # Checking Prediction
        predict = model.predict(testX[0].reshape(1, 28, 28, 1))
        pred_class = [np.argmax(i) for i in predict][0]
        print(f'\nOriginal : {testY[0]} | Prediction : {pred_class}\n')

        return True
    
    except Exception as ex:
        print(f'\nError! {ex}\n')
        return False

def start_recognizing(file_path: str) -> None:
    """
    Starts the recognition.

    file_path -> File of the trained model will load.

    Returns nothing.
    """
    try:
        global timeS, timeE
        # ========== Model Loading ==========
        try:
            model_nn = kr.models.load_model(file_path)
            print('\nModel loaded successfully!\n')
        except:
            print('\nError occurred during loading the model!\n')

        # ========== Start Camera ==========
        cap = start_cam()

        while cap.isOpened():
            _, image = cap.read()

            if not _: continue

            # ========== Preprocessing of captured image ==========
            height, width, channel = image.shape
            bbox = x, y, x + pad, y + pad
            image_crop = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            image_crop = cv2.rotate(image_crop, cv2.ROTATE_180)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            # image_crop = cv2.equalizeHist(image_crop)
            thrVal, image_crop = cv2.threshold(image_crop, 120, 255, cv2.THRESH_BINARY_INV)

            image_show = cv2.resize(image_crop, (200, 200))
            image_test = cv2.resize(image_crop, (28, 28))
            image_test = image_test / 255

            image[5: 205, width - 205: width - 5] = cv2.cvtColor(image_show, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(image, (5, 70), (width - 310, 135), RED, -1)
            cv2.rectangle(image, (width - 205, 210), (width - 5, height - 5), GREEN, -1)

            # ========== Model Prediction ==========
            try:
                if image_test.all() != 1.0:
                    pred = str(np.argmax(model_nn.predict(image_test.reshape(1, 28, 28, 1))))
                    accr = str(round(np.amax(model_nn.predict(image_test.reshape(1, 28, 28, 1))) * 100, 2))

                    cv2.putText(image, pred, (width - 160, 380), 1, 10, BLUE, 5)
                    cv2.putText(image, f'Accuracy : {accr}%', (15, 115), 3, 1, YELLOW, 2)
                    cv2.rectangle(image, (bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5), GREEN, 5)
                else:
                    cv2.putText(image, "No Image", (width - 180, 350), 1, 2, BLUE, 3)
                    cv2.putText(image, f'Accuracy : None', (15, 115), 3, 1, YELLOW, 2)
                    cv2.rectangle(image, (bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5), RED, 5)
            except Exception as ex:
                print(f'Error : {ex}')

            # ========== Calculating the FPS ==========
            timeE = t.time()
            fps = int(1 // (timeE - timeS))
            timeS = timeE
            cv2.rectangle(image, (5, 5), (165, 65), RED, -1)
            cv2.putText(image, str(f'FPS : {fps}'), (15, 45), 3, 1, YELLOW, 2)

            # ========== Display the Image ==========
            cv2.imshow('Digit Classifier - Chanchal Roy', image)

            if cv2.waitKey(1) & 0xff == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as ex: print(f'\nError! {ex}\n')

if __name__ == '__main__':
    option = int(input('\nTo Train the model enter 0 else enter 1 to start recognition : '))
    if option == 0: train = True
    else: train = False

    if train:
        start_training(myFile)
        train = False

    if not train:
        start_recognizing(myFile)
