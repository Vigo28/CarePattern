# Old main.py that tracks people inside zones.

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
hands_above_head_counts = {}  # {track_id: count}
active_bed_events = {}   # {track_id: ttl}
active_table_events = {}   # {track_id: ttl}
active_hands_events = {}  # {track_id: ttl}
hands_currently_up = {}  # Track current hand position state for each person
threshold_frames = 10

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

    # Check if keypoints and boxes exist before processing
    if results[0].keypoints is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        # Get keypoints for each person
        for person_keypoints, tid in zip(results[0].keypoints.data, results[0].boxes.id):
            tid = int(tid)  # Ensure tid is integer
            # Get relevant keypoint indices
            # 9: right wrist, 10: left wrist
            # 5: right shoulder, 6: left shoulder
            # 3: right ear, 4: left ear
            wrists = person_keypoints[[9, 10]]
            shoulders = person_keypoints[[5, 6]]
            ears = person_keypoints[[3, 4]]

            # Calculate average height of ears (approximate head position)
            head_y = (ears[0][1] + ears[1][1]) / 2
            # Calculate average height of shoulders
            shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2
            # Get wrist heights
            right_wrist_y = wrists[0][1]
            left_wrist_y = wrists[1][1]

            # Check if either hand is above head
            hands_up = (right_wrist_y < head_y or left_wrist_y < head_y) and \
                      right_wrist_y != 0 and left_wrist_y != 0

            # Only increment counter if hands weren't up before but are up now
            if hands_up and not hands_currently_up.get(tid, False):
                hands_above_head_counts[tid] = hands_above_head_counts.get(tid, 0) + 1
                active_hands_events[tid] = 50  # Set TTL for event display

                # Draw warning text with counter
                person_center = (int((shoulders[0][0] + shoulders[1][0]) / 2),
                               int(head_y - 30))
                cv2.putText(annotated, f"Hands up! ({hands_above_head_counts[tid]}x)",
                           person_center,
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)

            # Update the current state
            hands_currently_up[tid] = hands_up

    # Update the boxes check section too
    if results[0].boxes is not None and results[0].boxes.id is not None:
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

    # Display hands above head events
    for idx, tid in enumerate(list(active_hands_events.keys())):
        if text_y < box_y + box_height - 20:  # Check if text fits in box
            cv2.putText(annotated, f"Person {tid} raised hands {hands_above_head_counts[tid]}x",
                        (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 100, 100), 2)
            text_y += 25
        active_hands_events[tid] -= 1
        if active_hands_events[tid] <= 0:
            del active_hands_events[tid]

    cv2.imshow("Zones + Pose + Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()