ABOUT: "Real-time object approach detection through PyTorch's MiDaS monocular depth estimation and OpenCV (live video stream). Computes per-frame depth maps, smooths temporal noise, and detects approaching objects with spatial filtering + temporal confirmation."

Real-time object approach detection system using a webcam and monocular depth estimation from MiDaS.

Features
- Real-time webcam stream processing with OpenCV
- Monocular depth estimation using MiDaS small
- Temporal depth smoothing to reduce noise
- Adaptive tresholding using median absolute deviation
- Spatial filtering with waiting on center
- Temporal confirmation to reduce noise and false positives
- Displays and follows approaching object in red bounding box.

File structure
- main.py: main application loop
- midas_utils.py: loading MiDaS model + depth inference
- detection.py: approach detection logic
- config.py: parameters and thresholds list
- depthdisplay.py: side-by-side comparison of camera stream and MiDaS depth inference 
