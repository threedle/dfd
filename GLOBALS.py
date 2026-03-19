import numpy as np
import math

sourcecolor = np.array([4, 55, 242]) / 255
vectorcolor = np.array([54, 105, 242]) / 255
targetcolor = np.array([0, 255, 0]) / 255
constraintcolor = np.array([255, 0, 0]) / 255
defaultcolor = np.array([137,137,137]) / 255
bluecolor = np.array([54,94,137]) / 255

radius = 3.5
start_elev = math.radians(0)
end_elev = math.radians(30)
elev_samples = 2
start_azim = math.radians(0)
azim_samples = 8
end_azim = math.radians(360 - 360/azim_samples)

pointradius = 0.03
curveradius = 0.015

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

patch_sizes = {
    'diff3f': 14,
    'dino2': 14,
    'dino': 14,  # backward-compatible alias
    'dino3': 16,
    'clip': 14,
    'radio': 16,
    # Architecture-name aliases (so callers can key by actual model architecture/id)
    'dinov2_vitg14_reg': 14,
    'dinov3_vit7b16': 16,
}

SAM2_CHECKPOINT = "/net/projects/ranalab/guanzhi/DFD/sam2_repo/checkpoints/sam2.1_hiera_large.pt"