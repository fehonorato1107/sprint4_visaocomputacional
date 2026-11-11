import cv2
import numpy as np


def draw_boxes(img: np.ndarray, boxes: list, out_path: str,
               color=(0, 255, 0), thickness=2):
    """boxes: lista de (bbox_pts, texto, confiança)."""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for pts, text, conf in boxes:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(vis, [pts_np], True, color, thickness)

        x = int(pts_np[:, 0].min())
        y = max(int(pts_np[:, 1].min()) - 6, 12)
        label = f"{text} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x, y - th - 4), (x + tw + 4, y + 4),
                      color, -1)
        cv2.putText(vis, label, (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.imwrite(out_path, vis)
    return vis