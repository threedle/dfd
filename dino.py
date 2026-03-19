import torch
import numpy as np
from torchvision import transforms as tfs


patch_size = 14

def init_dino(device, repodir="facebookresearch/dinov2",
              archtype="dinov2_vitg14_reg", **kwargs):
    model = torch.hub.load(
        repodir, archtype, **kwargs
    )
    model = model.to(device).eval()
    return model

@torch.no_grad
def get_dino_features(device, dino_model, img, grid, normalize=True):
    transform = tfs.Compose(
        [
            tfs.Resize((518, 518)),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).permute(0, 2, 3, 1)

    if normalize:
        features = torch.nn.functional.normalize(features, dim=1) * 0.5

    return features

@torch.no_grad
def get_dino_features_batched(device, dino_model, imgs, grid, normalize=True,
                              batch_size=10, half=True, debug=False):
    from tqdm import tqdm
    transform = tfs.Compose(
        [
            tfs.Resize((518, 518)),
            # tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    grid = grid.repeat(len(batched_renderings), 1, 1, 1)

    ### Batch all the renders together ###
    from dino import get_dino_features_batched

    tot_aligned_features = torch.zeros(
            (len(batched_renderings), H, W, 768), device="cpu",
            dtype=torch.float16,
            )

    imgs = imgs[..., :3].permute(0, 3, 1, 2)
    features = []

    print(f"Getting DINO features for {imgs.shape} images ...")

    import time
    t0 = time.time()

    for i in tqdm(range(0, len(imgs), batch_size)):
        batch_imgs = transform(imgs[i:i + batch_size])
        # NOTE: This is the same as running forward_features() and extracting the patch tokens!!
        batch_features = dino_model.get_intermediate_layers(batch_imgs, n=1)[0]
        if half:
            batch_features = batch_features.half()
        h, w = int(batch_imgs.shape[2] / patch_size), int(batch_imgs.shape[3] / patch_size)
        dim = batch_features.shape[-1]
        batch_features = batch_features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
        batch_features = torch.nn.functional.grid_sample(
            batch_features, grid[i:i+batch_size], align_corners=False
        ).permute(0, 2, 3, 1) # B x H x W x C

        if normalize:
            batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

        features.append(batch_features)

    imgs = imgs.cpu()
    if debug:
        precattime = time.time()
        print(f"Time taken for DINO features: {precattime - t0:.2f} seconds")
    features = torch.cat(features, dim=0)

    if debug:
        postcattime = time.time()
        print(f"Time taken for concat: {postcattime - precattime:.2f} seconds")
        print(f"Total time taken for DINO features: {postcattime - t0:.2f} seconds")

    return features
