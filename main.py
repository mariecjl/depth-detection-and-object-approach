import cv2
import config
from midas_utils import load_midas, estimate_depth
from detection import detect_approach

#setup w midas and camera stream
midas, transform, device = load_midas()
cap = cv2.VideoCapture(0)

approach_counter = 0

#main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #estimating the depth
    depth = estimate_depth(frame, midas, transform, device)
    config.depth_buffer.append(depth)

    #buffer/waiting until enough past frames exist to compare depth changes
    if len(config.depth_buffer) < config.DEPTH_HISTORY:
        cv2.imshow("Approach Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    #approaching regions detection
    #contour and per-pixel depth change
    h, w, _ = frame.shape
    main_cnt, delta = detect_approach(
        depth, config.depth_buffer, (h, w), config
    )

    #binary mask of the detected contour and detection of the mean depth change
    if main_cnt is not None:
        mask = cv2.drawContours(
            np.zeros((h, w), dtype=np.uint8),
            [main_cnt], -1, 255, -1
        )
        mean_delta = delta[mask > 0].mean()
        if mean_delta > config.DELTA_HIGH:
            approach_counter += 1
    else:
        approach_counter = max(0, approach_counter - 2)

    #confirm approach over multiple frames
    if approach_counter >= config.APPROACH_FRAMES_REQUIRED and main_cnt is not None:
        x, y, bw, bh = cv2.boundingRect(main_cnt)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(
            frame, "APPROACHING OBJECT", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3
        )
        
    cv2.imshow("Approach Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
