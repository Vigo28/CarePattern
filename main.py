import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

pose_model = YOLO("yolov8n-pose.pt")

# Zone (voorbeeld)
bed_zone = Polygon([(24, 24), (360, 24), (360, 720), (24, 720)])
bed_zone = Polygon([(100, 200), (500, 200), (500, 400), (100, 400)])

in_bed_counts = {}
threshold_frames = 10
active_events = {}   # {track_id: ttl}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a named window and make it resizable
cv2.namedWindow("Zones + Pose + Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model.track(frame, persist=True, classes=[0], verbose=False)
    annotated = results[0].plot()

    if results[0].boxes.id is not None:
        for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tid = int(tid)

            # zone check
            if Point(cx, cy).within(bed_zone):
                in_bed_counts[tid] = in_bed_counts.get(tid, 0) + 1
                if in_bed_counts[tid] == threshold_frames:
                    # event met TTL van 50 frames (~2 sec bij 25fps)
                    active_events[tid] = 50
            else:
                in_bed_counts[tid] = 0

    # teken de bed-zone
    pts = [(int(x), int(y)) for x, y in bed_zone.exterior.coords]
    cv2.polylines(annotated, [np.array(pts)], isClosed=True, color=(255, 0, 0), thickness=2)

    # events tekenen en TTL aftellen
    for tid in list(active_events.keys()):
        cv2.putText(annotated, f"EVENT: Person {tid} in bed-zone",
                    (50, 50 + 30 * tid), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        active_events[tid] -= 1
        if active_events[tid] <= 0:
            del active_events[tid]

    cv2.imshow("Zones + Pose + Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()