# python
import threading
from pathlib import Path
import tempfile
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO

from .jobs import set_status, set_output, set_error, set_progress

def _process_video_file(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    set_status(job_id, "processing")
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temp file in same directory to write atomically
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=output_path.suffix, dir=str(output_path.parent))
    # Close the fd so Windows releases the handle before we unlink or let OpenCV create/write the file
    try:
        os.close(tmp_fd)
    except Exception:
        pass
    Path(tmp_path).unlink(missing_ok=True)  # remove created file so OpenCV can create it properly

    try:
        pose_model = YOLO(model_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open input video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video dimensions from input.")

        # Try common fourcc candidates and ensure writer opens
        fourcc_candidates = ("avc1", "mp4v", "H264")
        writer = None
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            if writer.isOpened():
                break
            else:
                try:
                    writer.release()
                except Exception:
                    pass

        if writer is None or not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter with available codecs.")

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = pose_model.track(frame, persist=True, classes=[0], verbose=False)
            except Exception:
                results = None

            annotated = frame
            if results and len(results) > 0:
                try:
                    plotted = results[0].plot()
                    if plotted is not None:
                        annotated = plotted
                except Exception:
                    annotated = frame

            # Normalize dtype to uint8
            if annotated is not None and annotated.dtype != np.uint8:
                # assume floats in range [0,1]
                annotated = np.clip(annotated * 255.0, 0, 255).astype(np.uint8)

            # If plotted likely in RGB (new array), convert to BGR for OpenCV writer
            if annotated is not frame and annotated.ndim == 3 and annotated.shape[2] == 3:
                try:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass

            # Ensure correct size
            if (annotated.shape[1], annotated.shape[0]) != (width, height):
                annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)

            frame_idx += 1
            if total_frames:
                percent = int(frame_idx * 100 / max(1, total_frames))
                set_progress(job_id, percent)

        # finalize
        cap.release()
        writer.release()
        try:
            pose_model.close()
        except Exception:
            pass

        # Move temp file into final location atomically
        shutil.move(str(tmp_path), str(output_path))

        set_output(job_id, str(output_path))
        set_status(job_id, "done")
    except Exception as e:
        # Ensure temp is removed on error
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        set_error(job_id, str(e))
    finally:
        # Extra safety: ensure handles released
        try:
            cap.release()
        except Exception:
            pass
        try:
            writer.release()
        except Exception:
            pass

def start_processing(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    t = threading.Thread(target=_process_video_file, args=(input_path, output_path, job_id, model_path), daemon=True)
    t.start()
    return t
