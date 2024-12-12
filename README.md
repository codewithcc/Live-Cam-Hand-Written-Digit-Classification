# Hand Digit Classifier

## Overview
The **Hand Digit Classifier** is a live digit recognition system that captures hand-drawn digits using OpenCV, processes them in real-time, and classifies them using a Convolutional Neural Network (CNN). This project leverages the power of machine learning libraries such as TensorFlow and Scikit-learn, alongside computer vision techniques provided by OpenCV.

## Features
- **Live Hand Digit Capture:** Utilizes a webcam to capture hand-drawn digits.
- **Real-Time Digit Detection:** Processes the captured frames in real-time for accurate digit classification.
- **CNN-based Classification:** Employs a Convolutional Neural Network built with TensorFlow for robust digit recognition.
- **Preprocessing with Scikit-learn:** Includes scaling, normalization, and preprocessing for optimal performance.

## Technologies Used
- **OpenCV:** For capturing and preprocessing live video footage.
- **TensorFlow:** For building and training the CNN model.
- **Scikit-learn:** For data preprocessing and evaluation metrics.
- **Python:** The primary programming language.

## Installation
Follow these steps to set up the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/codewithcc/hand-digit-classifier.git
   cd hand-digit-classifier
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.8 or higher installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset (Optional):**
   If you need to retrain the model, download the MNIST dataset or any other digit dataset of your choice.

4. **Run the Application:**
   Launch the live digit classifier:
   ```bash
   python main.py
   ```

## How It Works
1. **Live Capture:**
   - The webcam captures live footage.
   - Users can draw digits in the air or on a flat surface.

2. **Preprocessing:**
   - Frames are converted to grayscale.
   - The region of interest is extracted and resized to match the input shape of the CNN.

3. **Classification:**
   - The preprocessed digit is fed into the CNN model.
   - The model outputs the predicted digit with its probability.

4. **Output:**
   - The detected digit is displayed on the screen in real-time.

## CNN Architecture
- **Input Layer:** Accepts 28x28 grayscale images.
- **Convolutional Layers:** Extract features from the images.
- **Pooling Layers:** Downsample the feature maps.
- **Fully Connected Layers:** Map features to the digit classes.
- **Output Layer:** Predicts one of the 10 digit classes (0-9).

## Results
The classifier achieves an accuracy of **96%** on the MNIST test dataset and performs efficiently in real-time on live video.

## Future Enhancements
- **Enhance Detection:** Improve preprocessing for better detection of poorly drawn digits.
- **Add Gesture Support:** Allow users to draw digits using hand gestures without a physical medium.
- **Mobile Support:** Extend functionality to mobile devices for portable digit recognition.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request for any bugs or new features.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The creators of the **MNIST dataset**.
- OpenCV, TensorFlow, and Scikit-learn communities for their incredible tools and support.
