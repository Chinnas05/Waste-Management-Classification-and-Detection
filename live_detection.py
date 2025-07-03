import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('D:/Waste_Management/test_code/cam/waste_classifier.h5')

# Define categories (update with your 7 actual classes)
categories = ['glass', 'metal', 'organic', 'paper', 'plastic', 'cardboard', 'trash']  # Placeholder

# Image preprocessing function
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Open camera with mobile stream URL
cap = cv2.VideoCapture('http://192.168.31.97:8080/video')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    start_time = time.time()  # Start timing

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Downscale frame for faster processing (optional, adjust as needed)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Limit to top 3 largest contours to reduce processing
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:  # Minimum area
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            try:
                processed_roi = preprocess_image(roi)
                prediction = model.predict(processed_roi)
                category_idx = np.argmax(prediction)
                category = categories[category_idx]
                confidence = prediction[0][category_idx] * 100

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{category} ({confidence:.2f}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing ROI: {e}")

    # Show the frame
    cv2.imshow('Waste Classifier', frame)

    # Print processing time for debugging
    end_time = time.time()
    print(f"Frame processing time: {end_time - start_time:.2f} seconds")

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()