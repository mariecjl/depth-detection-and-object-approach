import cv2
import torch
import numpy as np

# Load MiDaS model from PyTorch Hub
model_type = "MiDaS_small"  # smaller version for faster real-time performance
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Load the preprocessing transforms from MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

# Open webcam (change index if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform, convert to batch format
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # Upsample to original frame resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    # Convert depth prediction to numpy
    depth_map = prediction.cpu().numpy()

    # Normalize depth map for display (0..255)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (255 * ((depth_map - depth_min) / (depth_max - depth_min))).astype(np.uint8)

    # Apply a colormap to make depth visually intuitive
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # Show camera feed and depth side by side
    combined = np.hstack((frame, depth_colored))
    cv2.imshow("Webcam & Depth Map", combined)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
