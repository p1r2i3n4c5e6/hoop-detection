from roboflow import Roboflow
import supervision as sv
import cv2

# Initialize Roboflow model
rf = Roboflow(api_key="MmDwODXTBoUQNSkBBwql")
project = rf.workspace().project("basketball-hooph")
model = project.version(1).model

# Set up annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set your desired downscale size (e.g., 416x416 or 320x320)
DOWNSCALE_WIDTH = 416
DOWNSCALE_HEIGHT = 416

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Downscale the frame for faster inference
    small_frame = cv2.resize(frame, (DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT))

    # Save the downscaled frame to a temporary file
    cv2.imwrite("temp.jpg", small_frame)

    # Run inference on the downscaled frame
    result = model.predict("temp.jpg", confidence=40, overlap=30).json()

    # Extract labels and detections
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)

    # Annotate the downscaled frame
    annotated_frame = bounding_box_annotator.annotate(
        scene=small_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    # Show annotated downscaled frame
    cv2.imshow("Roboflow Webcam Inference", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
