import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

pose_model = YOLO("yolo11n-pose.pt")

# Zone (voorbeeld)
bed_zone = Polygon([(0, 0), (820, 0), (820, 720), (0, 720)])
pts_bed_zone = [(int(x), int(y)) for x, y in bed_zone.exterior.coords]

table_zone = Polygon([(960, 0), (1280, 0), (1280, 720), (960, 720)])
pts_table_zone = [(int(x), int(y)) for x, y in table_zone.exterior.coords]

in_bed_counts = {}
in_table_counts = {}
threshold_frames = 10
active_bed_events = {}   # {track_id: ttl}
active_table_events = {}   # {track_id: ttl}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # use DirectShow on windows for fast ini
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a named window and make it resizable
cv2.namedWindow("Zones + Pose + Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the camera horizontally
    frame = cv2.flip(frame, 1)

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
                if in_bed_counts[tid] >= threshold_frames:
                    # Houd event actief zolang persoon in zone is
                    active_bed_events[tid] = 50
                    if in_bed_counts[tid] == threshold_frames:
                        print(f"Person {tid} in bed-zone")
            else:
                in_bed_counts[tid] = 0
                if tid in active_bed_events:
                    del active_bed_events[tid]

            if Point(cx, cy).within(table_zone):
                in_table_counts[tid] = in_table_counts.get(tid, 0) + 1
                if in_table_counts[tid] >= threshold_frames:
                    # Houd event actief zolang persoon in zone is
                    active_table_events[tid] = 50
                    if in_table_counts[tid] == threshold_frames:
                        print(f"Person {tid} at table-zone")
            else:
                in_table_counts[tid] = 0
                if tid in active_table_events:
                    del active_table_events[tid]

    # teken de zones
    cv2.polylines(annotated, [np.array(pts_bed_zone)], isClosed=True, color=(0, 0, 255), thickness=4)
    cv2.polylines(annotated, [np.array(pts_table_zone)], isClosed=True, color=(0, 255, 0), thickness=4)

    # Create fixed box for events display - centered at bottom
    frame_height, frame_width = annotated.shape[:2]
    box_width, box_height = 400, 200
    box_x = (frame_width - box_width) // 2  # Center horizontally
    box_y = frame_height - box_height - 20  # Position at bottom with margin
    
    # Draw semi-transparent background box
    overlay = annotated.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Draw box border
    cv2.rectangle(annotated, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)
    
    # Title
    cv2.putText(annotated, "ACTIVE EVENTS", (box_x + 10, box_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    text_y = box_y + 50
    
    # bed events tekenen en TTL aftellen
    for idx, tid in enumerate(list(active_bed_events.keys())):
        if text_y < box_y + box_height - 20:  # Check if text fits in box
            cv2.putText(annotated, f"Person {tid} in bed-zone",
                        (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 100, 255), 2)
            text_y += 25
        active_bed_events[tid] -= 1
        if active_bed_events[tid] <= 0:
            del active_bed_events[tid]

    # table events tekenen en TTL aftellen
    for idx, tid in enumerate(list(active_table_events.keys())):
        if text_y < box_y + box_height - 20:  # Check if text fits in box
            cv2.putText(annotated, f"Person {tid} at table-zone",
                        (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 255, 100), 2)
            text_y += 25
        active_table_events[tid] -= 1
        if active_table_events[tid] <= 0:
            del active_table_events[tid]

    cv2.imshow("Zones + Pose + Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()