from tensorflow import keras as kr
import numpy as np

if __name__ == '__main__':
    (trainX, trainY), (testX, testY) = kr.datasets.mnist.load_data()
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    # print(trainX[0].shape)

    trainX = trainX / 255
    testX = testX / 255
    trainX = trainX.reshape(len(trainX), 784)
    testX = testX.reshape(len(testX), 784)
    # print(trainX.shape, trainX[0].shape)
    # print(trainX[0])

    model = kr.Sequential([
        kr.layers.Dense(100, input_shape=(784,), activation='relu'),
        kr.layers.Dense(10, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(trainX, trainY, epochs=10)

    try:
        model.save(r'model_nn.h5')
        print('Model saved successfully!')
    except: print('Error occurred during saving the model!')

    model = kr.models.load_model(r'model_nn.h5')

    pred1 = model.predict(testX[0].reshape(1, 784))
    pred_class1 = np.argmax(pred1[0])
    print(f'\n-----Prediction of one data row-----\nOriginal Value : {testY[0]} | Prediction : {pred_class1}')

    pred2 = model.predict(testX)
    pred_class2 = np.argmax(pred2[0])
    print(f'\n-----Prediction of1st data row from all data row-----\nOriginal Value : {testY[0]} | Prediction : {pred_class2}')