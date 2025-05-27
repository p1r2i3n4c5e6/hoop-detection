from roboflow import Roboflow
import cv2

# Initialize Roboflow and load model
rf = Roboflow(api_key="MmDwODXTBoUQNSkBBwql")
project = rf.workspace().project("basketball-hooph")
model = project.version(1).model

# Read image
image = cv2.imread("test.jpg")

# Run prediction
result = model.predict("test.jpg", confidence=40, overlap=30).json()

# Draw bounding boxes and confidence
for pred in result["predictions"]:
    x1 = int(pred["x"] - pred["width"] / 2)
    y1 = int(pred["y"] - pred["height"] / 2)
    x2 = int(pred["x"] + pred["width"] / 2)
    y2 = int(pred["y"] + pred["height"] / 2)
    label = pred["class"]
    confidence = pred["confidence"]
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Put label and confidence
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show image in a window
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
