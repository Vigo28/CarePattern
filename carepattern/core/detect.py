from pathlib import Path

active_hands_events = {}
hands_above_head_counts = {}
hands_currently_up = {}
sitting_counts = {}
person_state = {}
active_sitting_events = {}
sitting_frame_counts = {}
lying_counts = {}
active_lying_events = {}

def format_timestamp(frame_number: int, fps: int) -> str:
    """Convert frame number to HH:MM:SS format."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def detect_hands_above_head(person_keypoints, tid, frame_number: int, output_path: Path, FPS: int):
    """Detect when hands are raised above head."""
    wrists = person_keypoints[[9, 10]]
    ears = person_keypoints[[3, 4]]

    head_y = (ears[0][1] + ears[1][1]) / 2
    right_wrist_y = wrists[0][1]
    left_wrist_y = wrists[1][1]

    hands_up = (right_wrist_y < head_y or left_wrist_y < head_y) and \
               right_wrist_y != 0 and left_wrist_y != 0

    if hands_up and not hands_currently_up.get(tid, False):
        hands_above_head_counts[tid] = hands_above_head_counts.get(tid, 0) + 1
        active_hands_events[tid] = 50
        timestamp = format_timestamp(frame_number, FPS)

        with open(output_path, 'a') as f:
            f.write(f'{timestamp} | Person {tid} | Hands above head, count:{hands_above_head_counts[tid]}\n')

    hands_currently_up[tid] = hands_up


def detect_sitting_on_bed(person_keypoints, tid, frame_number: int, output_path: Path, FPS: int):
    """Detect when person transitions from standing to sitting."""
    hips = person_keypoints[[11, 12]]
    knees = person_keypoints[[13, 14]]

    hip_y = (hips[0][1] + hips[1][1]) / 2
    knee_y = (knees[0][1] + knees[1][1]) / 2

    is_sitting = abs(hip_y - knee_y) < 75 and hip_y != 0 and knee_y != 0

    current_state = person_state.get(tid, 'standing')

    if is_sitting:
        sitting_frame_counts[tid] = sitting_frame_counts.get(tid, 0) + 1
    else:
        sitting_frame_counts[tid] = 0

    # Trigger only on transition from standing to sitting
    if sitting_frame_counts[tid] >= 3 and current_state != 'sitting':
        sitting_counts[tid] = sitting_counts.get(tid, 0) + 1
        active_sitting_events[tid] = 50
        timestamp = format_timestamp(frame_number, FPS)

        with open(output_path, 'a') as f:
            f.write(f'{timestamp} | Person {tid} | Sat down on bed, count:{sitting_counts[tid]}\n')

        person_state[tid] = 'sitting'
    elif not is_sitting:
        person_state[tid] = 'standing'


def detect_lying_on_bed(person_keypoints, tid, frame_number: int, output_path: Path, FPS: int):
    """Detect when person transitions from sitting to lying down on bed."""
    shoulders = person_keypoints[[5, 6]]
    hips = person_keypoints[[11, 12]]

    shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2
    hip_y = (hips[0][1] + hips[1][1]) / 2

    is_lying = abs(shoulder_y - hip_y) < 50 and shoulder_y != 0 and hip_y != 0

    current_state = person_state.get(tid, 'standing')

    if is_lying and current_state == 'sitting':
        lying_counts[tid] = lying_counts.get(tid, 0) + 1
        active_lying_events[tid] = 50
        timestamp = format_timestamp(frame_number, FPS)

        with open(output_path, 'a') as f:
            f.write(f'{timestamp} | Person {tid} | Lying down on bed, count:{lying_counts[tid]}\n')

        person_state[tid] = 'lying'
    elif not is_lying and current_state == 'lying':
        timestamp = format_timestamp(frame_number, FPS)
        with open(output_path, 'a') as f:
            f.write(f'{timestamp} | Person {tid} | Sat up from bed\n')
        person_state[tid] = 'sitting'


def process_datapoints(datapoints, frame_number: int, output_path: Path, fps: int):
    """Process all detected keypoints and apply detection rules."""
    if datapoints[0].keypoints is None or datapoints[0].boxes is None or datapoints[0].boxes.id is None:
        return

    for person_keypoints, tid in zip(datapoints[0].keypoints.data, datapoints[0].boxes.id):
        tid = int(tid)

        detect_hands_above_head(person_keypoints, tid, frame_number, output_path, fps)
        detect_sitting_on_bed(person_keypoints, tid, frame_number, output_path, fps)
        detect_lying_on_bed(person_keypoints, tid, frame_number, output_path, fps)
