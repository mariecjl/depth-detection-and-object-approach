from collections import deque

#thresholds and tunable values for later
DEPTH_HISTORY = 8
APPROACH_FRAMES_REQUIRED = 4
MIN_AREA_RATIO = 0.03
CENTER_WEIGHT_RADIUS = 0.40
DELTA_HIGH = 0.35
DELTA_LOW = 0.15

#depth buffer
depth_buffer = deque(maxlen=DEPTH_HISTORY)
