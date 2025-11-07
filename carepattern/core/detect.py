from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict
import math

job_persons: Dict[str, Dict[int, "Person"]] = defaultdict(dict)

# thresholds
KNEE_ANGLE_THRESHOLD_DEG = 150.0
HIP_KNEE_VERTICAL_THRESHOLD = 75.0
SHOULDER_HIP_VERTICAL_THRESHOLD = 50.0


def format_timestamp(frame_number: int, fps: float) -> str:
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# small safe extractors reused across the module
def _y_of(keypoints, idx: int) -> float:
    try:
        p = keypoints[idx]
        return float(p[1])
    except Exception:
        return 0.0


def _xy_of(keypoints, idx: int) -> Tuple[float, float]:
    try:
        p = keypoints[idx]
        return float(p[0]), float(p[1])
    except Exception:
        return 0.0, 0.0


def compute_knee_angle(keypoints) -> Optional[float]:
    """
    Compute the knee angle (hip - knee - ankle) in degrees using averaged left/right
    keypoints. Returns None if calculation isn't possible.
    """
    hip_x1, hip_y1 = _xy_of(keypoints, 11)
    hip_x2, hip_y2 = _xy_of(keypoints, 12)
    hip_x = (hip_x1 + hip_x2) / 2
    hip_y = (hip_y1 + hip_y2) / 2

    knee_x1, knee_y1 = _xy_of(keypoints, 13)
    knee_x2, knee_y2 = _xy_of(keypoints, 14)
    knee_x = (knee_x1 + knee_x2) / 2
    knee_y = (knee_y1 + knee_y2) / 2

    ankle_x1, ankle_y1 = _xy_of(keypoints, 15)
    ankle_x2, ankle_y2 = _xy_of(keypoints, 16)
    ankle_x = (ankle_x1 + ankle_x2) / 2
    ankle_y = (ankle_y1 + ankle_y2) / 2

    try:
        # vectors: hip->knee and ankle->knee
        v1x = hip_x - knee_x
        v1y = hip_y - knee_y
        v2x = ankle_x - knee_x
        v2y = ankle_y - knee_y
        dot = v1x * v2x + v1y * v2y
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 > 0 and n2 > 0:
            cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
            return math.degrees(math.acos(cosang))
    except Exception:
        pass
    return None


def posture_from_keypoints(keypoints) -> Tuple[float, float, float, Optional[float], bool, bool]:
    """
    Returns:
      hip_y, knee_y, shoulder_y, knee_angle_deg, is_sitting, is_lying
    """
    hip_y = (_y_of(keypoints, 11) + _y_of(keypoints, 12)) / 2
    knee_y = (_y_of(keypoints, 13) + _y_of(keypoints, 14)) / 2
    shoulder_y = (_y_of(keypoints, 5) + _y_of(keypoints, 6)) / 2

    knee_angle_deg = compute_knee_angle(keypoints)
    is_knee_bent = (knee_angle_deg is not None) and (knee_angle_deg < KNEE_ANGLE_THRESHOLD_DEG)

    is_sitting = ((abs(hip_y - knee_y) < HIP_KNEE_VERTICAL_THRESHOLD and hip_y != 0 and knee_y != 0)
                  or is_knee_bent)
    is_lying = (abs(shoulder_y - hip_y) < SHOULDER_HIP_VERTICAL_THRESHOLD and shoulder_y != 0 and hip_y != 0)

    return hip_y, knee_y, shoulder_y, knee_angle_deg, is_sitting, is_lying


@dataclass
class Person:
    tid: int
    state: str = "standing"
    sitting_frames: int = 0
    frames_not_sitting: int = 0
    sitting_count: int = 0
    lying_count: int = 0
    active_sitting_timer: int = 0
    active_lying_timer: int = 0

    def tick(self):
        if self.active_sitting_timer > 0:
            self.active_sitting_timer -= 1
        if self.active_lying_timer > 0:
            self.active_lying_timer -= 1

    def _log_transition(self, old_state: str, new_state: str, frame_number: int, output_path: Path, fps: float, extra: str = ""):
        ts = format_timestamp(frame_number, fps)
        msg = f"{ts} | Person {self.tid} | {old_state} -> {new_state}"
        if extra:
            msg += f" | {extra}"
        with open(output_path, "a") as f:
            f.write(msg + "\n")

    def update(self, keypoints, frame_number: int, output_path: Path, fps: float):
        hip_y, knee_y, shoulder_y, knee_angle_deg, is_sitting, is_lying = posture_from_keypoints(keypoints)

        threshold_frames = max(3, int(fps * 0.4))
        cooldown_frames = int(fps * 2)

        # update frame counters
        if is_sitting:
            self.frames_not_sitting = 0
        else:
            self.frames_not_sitting += 1

        # State machine transitions (only allowed transitions)
        old_state = self.state

        if self.state == "standing":
            if is_sitting:
                self.sitting_frames += 1
            else:
                self.sitting_frames = 0

            if self.sitting_frames >= threshold_frames and self.active_sitting_timer == 0:
                # standing -> sitting
                self.sitting_count += 1
                self.active_sitting_timer = cooldown_frames
                extra = f"Sat down, count:{self.sitting_count}"
                if knee_angle_deg is not None:
                    extra += f", knee_angle:{knee_angle_deg:.1f}"
                self._log_transition(old_state, "sitting", frame_number, output_path, fps, extra)
                self.state = "sitting"

        elif self.state == "sitting":
            if is_lying and self.active_lying_timer == 0:
                # sitting -> lying (instant)
                self.lying_count += 1
                self.active_lying_timer = cooldown_frames
                self._log_transition(old_state, "lying", frame_number, output_path, fps, f"Lying down, count:{self.lying_count}")
                self.state = "lying"
                # reset sitting trackers
                self.sitting_frames = 0
                self.frames_not_sitting = 0
            else:
                # require a short consecutive not-sitting to consider stood up
                if not is_sitting and self.frames_not_sitting >= threshold_frames:
                    # sitting -> standing
                    self._log_transition(old_state, "standing", frame_number, output_path, fps, "Stood up")
                    self.state = "standing"
                    self.sitting_frames = 0
                    self.frames_not_sitting = 0
                elif is_sitting:
                    # remain sitting, ensure counters are up-to-date
                    self.sitting_frames += 1

        elif self.state == "lying":
            # only allowed to go from lying -> sitting
            if not is_lying and is_sitting:
                # lying -> sitting
                # increment sitting count because coming to sitting is considered a sit event
                self.sitting_count += 1
                self.active_sitting_timer = cooldown_frames
                self._log_transition(old_state, "sitting", frame_number, output_path, fps, f"Sat up from lying, count:{self.sitting_count}")
                self.state = "sitting"
                # reset counters
                self.sitting_frames = threshold_frames  # treat as already sitting for continuity
                self.frames_not_sitting = 0
            # otherwise stay lying until a sitting posture is detected

        return


def process_datapoints(datapoints, frame_number: int, output_path: Path, fps: float, job_id: str):
    # quick sanity
    if datapoints[0].keypoints is None or datapoints[0].boxes is None or datapoints[0].boxes.id is None:
        return

    persons = job_persons[job_id]

    # tick all known persons for this job to decrement cooldowns
    for p in persons.values():
        p.tick()

    # compute detection thresholds once per frame
    threshold_frames = max(3, int(fps * 0.4))

    for keypoints, tid_raw in zip(datapoints[0].keypoints.data, datapoints[0].boxes.id):
        tid = int(tid_raw)
        person = persons.get(tid)
        if person is None:
            hip_y, knee_y, shoulder_y, knee_angle_deg, is_sitting, is_lying = posture_from_keypoints(keypoints)

            # choose initial state from first detection to avoid logging a spurious transition
            if is_lying:
                init_state = "lying"
                init_sitting_frames = 0
            elif is_sitting:
                init_state = "sitting"
                init_sitting_frames = threshold_frames  # treat as already stable sitting
            else:
                init_state = "standing"
                init_sitting_frames = 0

            person = Person(tid=tid, state=init_state, sitting_frames=init_sitting_frames)
            persons[tid] = person

        person.update(keypoints, frame_number, output_path, fps)