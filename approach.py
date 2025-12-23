import cv2
import torch
import numpy as np
from collections import deque

# ------------------ MiDaS setup ------------------
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

cap = cv2.VideoCapture(0)

# ------------------ Stabilization buffers ------------------
DEPTH_HISTORY = 6
depth_buffer = deque(maxlen=DEPTH_HISTORY)

APPROACH_FRAMES_REQUIRED = 4
approach_counter = 0

# ------------------ Tunables ------------------
DELTA_THRESHOLD = 0.2
MIN_AREA_RATIO = 0.02       # fraction of image area
CENTER_WEIGHT_RADIUS = 0.35 # center region (fraction)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = depth.cpu().numpy()
    depth = cv2.GaussianBlur(depth, (7, 7), 0)

    depth_buffer.append(depth)

    if len(depth_buffer) < DEPTH_HISTORY:
        cv2.imshow("Approach Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    prev_depth = np.mean(list(depth_buffer)[:-1], axis=0)
    delta = depth - prev_depth

    # ------------------ Spatial filtering ------------------
    # Center mask
    cx, cy = w // 2, h // 2
    rx, ry = int(w * CENTER_WEIGHT_RADIUS), int(h * CENTER_WEIGHT_RADIUS)

    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[cy - ry:cy + ry, cx - rx:cx + rx] = 1

    # Only consider close & centered regions
    close_mask = depth > np.percentile(depth, 75)
    approach_mask = (delta > DELTA_THRESHOLD) & close_mask & center_mask
    approach_mask = approach_mask.astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    approach_mask = cv2.morphologyEx(approach_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        approach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ------------------ Choose dominant region ------------------
    main_cnt = None
    max_area = 0
    min_area = MIN_AREA_RATIO * (h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area > max_area:
            max_area = area
            main_cnt = cnt

    # ------------------ Temporal consistency ------------------
    if main_cnt is not None:
        approach_counter += 1
    else:
        approach_counter = max(0, approach_counter - 1)

    if approach_counter >= APPROACH_FRAMES_REQUIRED:
        x, y, bw, bh = cv2.boundingRect(main_cnt)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(
            frame,
            "APPROACHING OBJECT",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Approach Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
