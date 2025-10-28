import threading
from typing import Optional, Dict, Any

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Optional[Any]]] = {}

def create_job() -> str:
    import uuid
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"status": "pending", "output": None, "error": None, "progress": 0}
    return job_id

def get_job(job_id: str) -> Optional[Dict[str, Optional[Any]]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None

def set_status(job_id: str, status: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = status

def set_output(job_id: str, output_path: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["output"] = output_path

def set_error(job_id: str, error_msg: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = error_msg

def set_progress(job_id: str, progress: int) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = progress