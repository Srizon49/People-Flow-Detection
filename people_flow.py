from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("people-walking.mp4")

if not cap.isOpened():
    print("Video open failed")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "tracked_people_flow.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

tracker = sv.ByteTrack()

heatmap = np.zeros((height, width), dtype=np.float32)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    detections = detections[detections.class_id == 0]

    detections = tracker.update_with_detections(
        detections
    )

    if detections.tracker_id is not None:

        for i in range(len(detections.xyxy)):

            x1, y1, x2, y2 = map(
                int,
                detections.xyxy[i]
            )

            track_id = int(
                detections.tracker_id[i]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.circle(
                heatmap,
                (center_x, center_y),
                15,
                1,
                -1
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    out.write(frame)

cap.release()
out.release()

heatmap_norm = cv2.normalize(
    heatmap,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

heatmap_norm = heatmap_norm.astype(
    np.uint8
)

heatmap_img = cv2.applyColorMap(
    heatmap_norm,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    "people_heatmap.png",
    heatmap_img
)

print("Finished")