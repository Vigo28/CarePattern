# python
import threading
from pathlib import Path
import cv2
import numpy as np
import av
from ultralytics import YOLO

from .jobs import set_status, set_output, set_error, set_progress

def _process_video_file(input_path: str, output_path: str, skeleton_output_path, job_id: str, model_path: str = "yolo11n-pose.pt"):
    set_status(job_id, "processing")
    input_path = Path(input_path)
    output_path = Path(output_path)
    skeleton_output_path = Path(skeleton_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = None
    pose_model = None

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

        container_overlay = av.open(str(output_path), mode="w")
        stream_overlay = container_overlay.add_stream("libx264", rate=int(round(fps)))
        stream_overlay.width = width
        stream_overlay.height = height
        stream_overlay.pix_fmt = "yuv420p"
        stream_overlay.options = {"preset": "veryfast", "crf": "23"}

        container_skeleton = av.open(str(skeleton_output_path), mode="w")
        stream_skeleton = container_skeleton.add_stream("libx264", rate=int(round(fps)))
        stream_skeleton.width = width
        stream_skeleton.height = height
        stream_skeleton.pix_fmt = "yuv420p"
        stream_skeleton.options = {"preset": "veryfast", "crf": "23"}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame
            skeleton_only = np.zeros_like(frame)

            try:
                results = pose_model.track(frame, persist=True, classes=[0], verbose=False)
            except Exception:
                results = None

            if results and len(results) > 0:
                try:
                    plotted = results[0].plot()
                    if isinstance(plotted, np.ndarray) and plotted.size:
                        annotated = plotted
                    # Draw skeleton on blank frame
                    skeleton_only = results[0].plot(img=skeleton_only)
                except Exception:
                    annotated = frame
                    skeleton_only = np.zeros_like(frame)
            else:
                skeleton_only = np.zeros_like(frame)

            # Ensure uint8 and correct size
            for img in [annotated, skeleton_only]:
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if getattr(img, "max", lambda: 255)() <= 1.0 else img.astype(np.uint8)
                if (img.shape[1], img.shape[0]) != (width, height):
                    img = cv2.resize(img, (width, height))

            # Encode and mux overlay video
            video_frame_overlay = av.VideoFrame.from_ndarray(annotated, format="bgr24")
            for packet in stream_overlay.encode(video_frame_overlay):
                container_overlay.mux(packet)

            # Encode and mux skeleton-only video
            video_frame_skeleton = av.VideoFrame.from_ndarray(skeleton_only, format="bgr24")
            for packet in stream_skeleton.encode(video_frame_skeleton):
                container_skeleton.mux(packet)

            frame_idx += 1
            if total_frames:
                percent = int(frame_idx * 100 / max(1, total_frames))
                set_progress(job_id, percent)

        # Flush encoders and close containers
        for packet in stream_overlay.encode():
            container_overlay.mux(packet)
        container_overlay.close()

        for packet in stream_skeleton.encode():
            container_skeleton.mux(packet)
        container_skeleton.close()

        set_output(job_id, str(output_path))
        set_status(job_id, "done")
    except Exception as e:
        set_error(job_id, str(e))
        try:
            if container_overlay is not None:
                container_overlay.close()
            if container_skeleton is not None:
                container_skeleton.close()
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

def start_processing(input_path: str, output_path: str, skeleton_output_path: str, job_id: str, model_path: str = "yolo11n-pose.pt"):
    t = threading.Thread(target=_process_video_file, args=(input_path, output_path, skeleton_output_path, job_id, model_path), daemon=True)
    t.start()
    return t
