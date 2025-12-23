import torch
import cv2
import numpy as np

def load_midas():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    return midas, transform, device


@torch.no_grad()
def estimate_depth(frame, midas, transform, device):
    h, w, _ = frame.shape

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    depth = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth = depth.cpu().numpy()
    depth = cv2.GaussianBlur(depth, (7, 7), 0)

    return depth
