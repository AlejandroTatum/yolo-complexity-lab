from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


def demo_frame(size: int = 640):
    """Create a deterministic RGB demo image without external assets."""
    import cv2
    import numpy as np

    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (60, 120), (250, 430), (60, 120, 220), -1)
    cv2.rectangle(img, (340, 180), (560, 390), (80, 180, 120), -1)
    cv2.circle(img, (465, 285), 60, (230, 120, 80), -1)
    cv2.putText(img, "YOLO Complexity Lab", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(img, "demo frame", (360, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return img


def read_image_file(file: BinaryIO | str | Path):
    import cv2
    import numpy as np

    if isinstance(file, (str, Path)):
        data = Path(file).read_bytes()
    else:
        data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo leer la imagen.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frames_from_video(path: str | Path, limit: int, stride: int = 1) -> list[object]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    index = 0
    try:
        while len(frames) < limit:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % max(1, stride) == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        cap.release()
    return frames


def frames_from_webcam(camera_index: int, limit: int) -> list[object]:
    import cv2

    cap = cv2.VideoCapture(camera_index)
    frames = []
    try:
        while len(frames) < limit:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def repeat_frame(frame: object, count: int) -> list[object]:
    return [frame for _ in range(max(1, count))]
