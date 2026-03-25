## Code for automatic generation of segmentation masks for input image w/ merging
import torch
import cv2
import numpy as np
from PIL import Image

def extra_merging(masks, iou_thresh=0.7):
    def mask_iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 0

    merged = []

    for m in masks:
        keep = True
        for mm in merged:
            if mask_iou(m["segmentation"], mm["segmentation"]) > iou_thresh:
                keep = False
                break
        if keep:
            merged.append(m)

    masks = merged

    return masks

def show_anns(anns, borders=True):
    import matplotlib.pyplot as plt

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

def main(img: np.ndarray, patch_size=16):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from GLOBALS import SAM2_CHECKPOINT
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = build_sam2(
        model_cfg,
        SAM2_CHECKPOINT,
        device=device,
        apply_postprocessing=False
    )

    points_per_side = img.shape[0] // patch_size
    mask_generator = SAM2AutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,          # 1 point per patch
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                use_m2m=True,
                box_nms_thresh=0.7,         # controls mask merging
                # crop_n_layers=1,            # check segmentations for small-scale render structures
            )

    masks = mask_generator.generate(img)

    return masks

if __name__ == "__main__":
    image = Image.open('images/cars.jpg')
    image = np.array(image.convert("RGB"))

    main(image)