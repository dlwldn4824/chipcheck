from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraStream:
    """Phase 5: 카메라 → 전처리 → 추론 → 오버레이. 현재는 스켈레톤."""

    source: int | str = 0
    fps_target: int = 15

    def frames(self):
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python-headless가 필요합니다.") from e

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패: {self.source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
