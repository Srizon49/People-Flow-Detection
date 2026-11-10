from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video path
video_path = "people-walking.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video open failed!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output_people_flow.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Person",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    out.write(frame)

cap.release()
out.release()

print("Done!")