import os
import cv2
import numpy as np




# Function to load and preprocess images
def load_dataset(dataset_path):
    X = []  # List to store preprocessed images
    y = []  # List to store labels (0 for not happy, 1 for happy)

    # Iterate through each subfolder in the dataset
    for label, emotion in enumerate(["not_happy", "happy","angry","sad","surprise","fear","disgust"]):
        emotion_path = os.path.join(dataset_path, emotion)

        # Iterate through each image file in the subfolder
        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)

            # Read and preprocess the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            image = cv2.resize(image, (48, 48))  # Resize image to 48x48 pixels
            image = image.astype(np.float32) / 255.0  # Normalize pixel values
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y)


# Path to the dataset folder
dataset_path = "emotions"

# Load and preprocess the dataset
X, y = load_dataset(dataset_path)

# Split the dataset into training and testing sets (e.g., 80% for training, 20% for testing)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.utils import to_categorical

# Perform one-hot encoding on the target labels
y_train_encoded = to_categorical(y_train, num_classes=7)
y_test_encoded = to_categorical(y_test, num_classes=7)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Output layer with 5 nodes (one for each emotion)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()




# Reshape the input data to add a channel dimension (for grayscale images)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# # Train the model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Function to detect faces and classify emotions
emotions = ["not_happy", "happy", "angry", "sad", "surprise","fear","disgust"]
def detect_emotion(model, face_cascade):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each detected face, classify emotion
        for (x, y, w, h) in faces:

            # Extract face region from the frame
            face_roi = gray[y:y+h, x:x+w]

            # Resize face region to 48x48 pixels (same size as training images)
            face_roi_resized = cv2.resize(face_roi, (48, 48))

            # Preprocess the face region
            face_roi_resized = face_roi_resized.astype(np.float32) / 255.0
            face_roi_resized = np.expand_dims(face_roi_resized, axis=-1)

            # Classify emotion using the model
            prediction = model.predict(np.array([face_roi_resized]))
            predicted_emotion_index = np.argmax(prediction[0])

            emotion_label = emotions[predicted_emotion_index]
            # label = "Happy" if prediction[0][0] > 0.5 else "Not Happy"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Load pre-trained face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Call the function to detect emotion using webcam
detect_emotion(model, face_cascade)