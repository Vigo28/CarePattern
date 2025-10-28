# python
import threading
from pathlib import Path
import cv2
import numpy as np
import av
from ultralytics import YOLO

from .jobs import set_status, set_output, set_error, set_progress

def _process_video_file(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    set_status(job_id, "processing")
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = None
    pose_model = None
    container = None
    stream = None

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

        # Open output container and add H264 stream (encoded via libx264 in the underlying libav)
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # optional encoder options
        stream.options = {"preset": "veryfast", "crf": "23"}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame
            try:
                results = pose_model.track(frame, persist=True, classes=[0], verbose=False)
            except Exception:
                results = None

            if results and len(results) > 0:
                try:
                    plotted = results[0].plot()
                    if isinstance(plotted, np.ndarray) and plotted.size:
                        annotated = plotted
                except Exception:
                    annotated = frame

            if annotated is None:
                annotated = frame

            # Ensure uint8 and correct size
            if annotated.dtype != np.uint8:
                # simple conversion assuming values in 0..1 or numeric
                annotated = (np.clip(annotated, 0, 1) * 255).astype(np.uint8) if getattr(annotated, "max", lambda: 255)() <= 1.0 else annotated.astype(np.uint8)

            if (annotated.shape[1], annotated.shape[0]) != (width, height):
                annotated = cv2.resize(annotated, (width, height))

            # Convert numpy BGR image into an av.VideoFrame and encode
            # Using format 'bgr24' because OpenCV frames are BGR
            video_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

            frame_idx += 1
            if total_frames:
                percent = int(frame_idx * 100 / max(1, total_frames))
                set_progress(job_id, percent)

        # flush encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        set_output(job_id, str(output_path))
        set_status(job_id, "done")
    except Exception as e:
        set_error(job_id, str(e))
        try:
            if container is not None:
                container.close()
        except Exception:
            pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if pose_model is not None:
                pose_model.close()
        except Exception:
            pass

def start_processing(input_path: str, output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    t = threading.Thread(target=_process_video_file, args=(input_path, output_path, job_id, model_path), daemon=True)
    t.start()
    return t
