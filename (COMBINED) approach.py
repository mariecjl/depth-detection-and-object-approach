import cv2
import torch
import numpy as np
from collections import deque

#midas set up
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

#loading transforms for midas small
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

cap = cv2.VideoCapture(0)

#depth history to smooth out noise from backgrounds
DEPTH_HISTORY = 8
depth_buffer = deque(maxlen=DEPTH_HISTORY)

APPROACH_FRAMES_REQUIRED = 4
approach_counter = 0

#defining the min amount of area needed (to avoid small fluctuations)
#defining the central part of the frame where tracking occurs
MIN_AREA_RATIO = 0.03
CENTER_WEIGHT_RADIUS = 0.40

#thresholds to count frame as approaching -- min 0.35 to count as approaching, max 0.15 otherwise
DELTA_HIGH = 0.35
DELTA_LOW = 0.15

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    #converting to timm accepted coloration
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    #applying midas
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    #blurring the depth map to avoid noise and sudden jumps
    depth = depth.cpu().numpy()
    depth = cv2.GaussianBlur(depth, (7, 7), 0)

    depth_buffer.append(depth)

    #"warm up" before detection occurs
    if len(depth_buffer) < DEPTH_HISTORY:
        cv2.imshow("Approach Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    #calc depth change
    prev_depth = np.mean(list(depth_buffer)[:-1], axis=0)
    delta = depth - prev_depth

    # adaptive threshold
    #makes the model more robust against noise
    delta_med = np.median(delta)
    delta_mad = np.median(np.abs(delta - delta_med)) + 1e-6
    ADAPTIVE_THRESHOLD = delta_med + 2.0 * delta_mad

    # blur for spacial consistency
    #uses this to filter out any potential noise
    delta_smooth = cv2.GaussianBlur(delta, (15, 15), 0)

    # mask
    # center gating -- defining center of the image
    cx, cy = w // 2, h // 2
    rx, ry = int(w * CENTER_WEIGHT_RADIUS), int(h * CENTER_WEIGHT_RADIUS)

    #creating the center mask
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[cy - ry:cy + ry, cx - rx:cx + rx] = 1

    # ignoring the background
    depth_near = depth > np.percentile(depth, 70)

    #combining all conditions for the detection of an approaching object
    approach_mask = (
        (delta > ADAPTIVE_THRESHOLD) &
        (delta_smooth > ADAPTIVE_THRESHOLD * 0.8) &
        depth_near &
        center_mask
    ).astype(np.uint8) * 255

    #connects broken regions and removes tiny gaps
    kernel = np.ones((7, 7), np.uint8)
    approach_mask = cv2.morphologyEx(approach_mask, cv2.MORPH_CLOSE, kernel)

    #contor detection -- object extraction
    contours, _ = cv2.findContours(
        approach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    #selecting the dominant region (largest approaching object)
    main_cnt = None
    max_area = 0
    min_area = MIN_AREA_RATIO * (h * w)

    #detecting the largest region by area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area > max_area:
            max_area = area
            main_cnt = cnt

    #temporal confirmation
    #consistently moving towards camera strongly enough to count
    if main_cnt is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [main_cnt], -1, 255, -1)
        mean_delta = np.mean(delta[mask > 0])
        #only if the confidence is high enough for the approach
        if mean_delta > DELTA_HIGH:
            approach_counter += 1
    else:
        approach_counter = max(0, approach_counter - 2)

    # visualization and display in the frame
    if approach_counter >= APPROACH_FRAMES_REQUIRED and main_cnt is not None:
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
