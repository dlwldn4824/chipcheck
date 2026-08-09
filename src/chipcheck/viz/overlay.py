from __future__ import annotations

from typing import Any


def draw_prediction(frame, prediction: Any, color=(0, 200, 80)):
    """판정 결과를 프레임에 표시."""
    import cv2

    label = getattr(prediction, "label", str(prediction))
    conf = getattr(prediction, "confidence", None)
    text = f"{label}" if conf is None else f"{label} {conf:.2f}"
    if getattr(prediction, "needs_review", False):
        text += " [REVIEW]"
        color = (0, 165, 255)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame
