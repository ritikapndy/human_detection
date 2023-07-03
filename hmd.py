#import libary
import cv2

# Load the pre-trained model files
model_weights = 'C:\\Users\\KIIT\\Downloads\\Programs\\python\\MobileNetSSD_deploy.caffemodel'
model_config = 'C:\\Users\\KIIT\\Downloads\\Programs\\python\\MobileNetSSD_deploy.prototxt.txt'
confidence_threshold = 0.5

# Load the model
net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for faster processing (optional)
    frame = cv2.resize(frame, (300, 300))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network
    detections = net.forward()

    # Process the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter detections by confidence threshold
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            # Check if the detected object is a person
            if class_id == 15:
                # Get the coordinates of the detection
                x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * frame.shape[0])

                # Draw a bounding box around the person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Human Detection", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
