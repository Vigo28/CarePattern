import threading
from pathlib import Path
import cv2
from ultralytics import YOLO

from .jobs import set_status, set_output, set_error, set_progress

def _process_video_file(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    set_status(job_id, "processing")
    try:
        pose_model = YOLO(model_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open input video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pose_model.track(frame, persist=True, classes=[0], verbose=False)

            annotated = frame
            if results and len(results) > 0:
                try:
                    plotted = results[0].plot()
                    if plotted is not None:
                        annotated = plotted
                except Exception:
                    pass

            if (annotated.shape[1], annotated.shape[0]) != (width, height):
                annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)

            frame_idx += 1
            if total_frames:
                percent = int(frame_idx * 100 / max(1, total_frames))
                set_progress(job_id, percent)

        cap.release()
        writer.release()
        try:
            pose_model.close()
        except Exception:
            pass

        set_output(job_id, str(output_path))
        set_status(job_id, "done")
    except Exception as e:
        set_error(job_id, str(e))

def start_processing(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    t = threading.Thread(target=_process_video_file, args=(input_path, output_path, job_id, model_path), daemon=True)
    t.start()
    return t