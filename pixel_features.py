import torch
import numpy as np
from tqdm import tqdm
from time import time
import random
from GLOBALS import patch_sizes

def expand_patches(down, patch_size=16):
    """
    down: (B', H', W', C') or (B, H', W')
    Upsamples by repeating each patch patch_size times in H and W dimensions.
    """
    if down.ndim == 4:
        # (B, H', W', C) -> use F.interpolate for efficiency
        return torch.nn.functional.interpolate(
            down.permute(0, 3, 1, 2),
            scale_factor=patch_size,
            mode='nearest',
        ).permute(0, 2, 3, 1)
    elif down.ndim == 3:
        # (B, H', W') -> use F.interpolate
        return torch.nn.functional.interpolate(
            down.unsqueeze(1).float(),
            scale_factor=patch_size,
            mode='nearest',
        ).squeeze(1).to(down.dtype)
    else:
        return torch.repeat_interleave(
            torch.repeat_interleave(down, patch_size, dim=1),
            patch_size, dim=2
        )

# Computes cosine distance between 4 neighbor features (TBLR -- zero out TB or LR if any are empty patches)
def neighbor_cosine(features, patchmask):
    device = features.device
    B, H, W, C = features.shape

    # 4 neighbor offsets: N, S, W, E
    shifts = [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
    ]

    # Shift features and masks
    neighbors = torch.stack(
        [torch.roll(features, shifts=(di, dj), dims=(1, 2)) for di, dj in shifts],
        dim=3
    )  # (B, H, W, 4, C)

    neighbor_masks = torch.stack(
        [torch.roll(patchmask, shifts=(di, dj), dims=(1, 2)) for di, dj in shifts],
        dim=3
    )  # (B, H, W, 4)

    # Invalidate wrapped borders
    for k, (di, dj) in enumerate(shifts):
        if di == 1:
            neighbor_masks[:, 0, :, k] = False
        if di == -1:
            neighbor_masks[:, -1, :, k] = False
        if dj == 1:
            neighbor_masks[:, :, 0, k] = False
        if dj == -1:
            neighbor_masks[:, :, -1, k] = False

    # Mask out all of N/S or E/W if just one is masked in the pair
    neighbor_masks &= neighbor_masks[..., [1,0,3,2]]

    # Only consider neighbors if center pixel is valid
    neighbor_masks &= patchmask.unsqueeze(-1)

    # Broadcasted cosine distances
    # NOTE: masked neighbors get cosine distance of 0 => the wipe never starts here!
    similarities = torch.zeros(neighbor_masks.shape, device=neighbor_masks.device, dtype=torch.half)
    similarities[neighbor_masks] = 1 - torch.sum(torch.repeat_interleave(features.unsqueeze(3), 4, dim=3)[neighbor_masks] * neighbors[neighbor_masks], -1) # (B, H, W, 4)

    return similarities

# NOTE: Written by chatgpt but looks good (I manually added batch dim)
# Computes sum of per-channel variance over 8-connected neighbors.
def neighbor_variance_8ring(features, patchmask):
    """
    Compute sum of per-channel variance over 8-connected neighbors.
    """

    device = features.device
    B, H, W, C = features.shape

    # 8 neighbor offsets: N, S, W, E, NW, NE, SW, SE
    shifts = [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
        ( 1,  1), ( 1, -1), (-1,  1), (-1, -1),
    ]

    # Shift features and masks
    neighbors = torch.stack(
        [torch.roll(features, shifts=(di, dj), dims=(1, 2)) for di, dj in shifts],
        dim=0
    )  # (8, B, H, W, C)

    neighbor_masks = torch.stack(
        [torch.roll(patchmask, shifts=(di, dj), dims=(1, 2)) for di, dj in shifts],
        dim=0
    )  # (8, B, H, W)

    # Invalidate wrapped borders
    for k, (di, dj) in enumerate(shifts):
        if di == 1:
            neighbor_masks[k, :, 0, :] = False
        if di == -1:
            neighbor_masks[k, :, -1, :] = False
        if dj == 1:
            neighbor_masks[k, :, :, 0] = False
        if dj == -1:
            neighbor_masks[k, :, :, -1] = False

    # Only consider neighbors if center pixel is valid
    neighbor_masks &= patchmask.unsqueeze(0)

    # Expand mask to channels
    neighbor_masks_c = neighbor_masks.unsqueeze(-1)  # (8, B, H, W, 1)

    # Count valid neighbors
    counts = neighbor_masks.sum(dim=0)  # (B, H, W)
    safe_counts = counts.clamp(min=1)

    # Mean
    mean = (neighbors * neighbor_masks_c).sum(dim=0) / safe_counts.unsqueeze(-1)

    # Variance
    var = (
        (neighbors - mean.unsqueeze(0)) ** 2
        * neighbor_masks_c
    ).sum(dim=0) / safe_counts.unsqueeze(-1)

    # Sum over channels
    variance = var.sum(dim=-1) # (B, H, W)

    # Zero where no neighbors
    variance = variance * (counts > 0)

    return variance

def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled

def get_pixel_features_diff3f(
    device,
    pipe,
    dino_model,
    prompt,
    renderbatch,
    H=512,
    W=512,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    return_image=True,
    prompts_list=None,
    normalize=True,
    debug=False,
    batch_size=10,
):
    """ Returns B x H x W x C features for each pixel in the renders"""
    from torchvision import transforms as tfs
    t1 = time()

    patch_size = patch_sizes['diff3f']

    # Load the batched renders from input tuple
    batched_renderings, normal_batched_renderings, depth = renderbatch

    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()

    batched_renderings = batched_renderings.cpu()
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    grid = grid.repeat(len(batched_renderings), 1, 1, 1)

    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()

    ### Batch all the renders together ###
    from diffusion import run_diffusion_batched

    tot_aligned_features = torch.zeros(
            (len(batched_renderings), H, W, 2048), device="cpu",
            dtype=torch.float16,
            )

    for i in tqdm(range(0, len(batched_renderings), batch_size)):
        predifftime = time()

        diffusion_input_img = (
            batched_renderings[i:i+batch_size, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)

        if use_normal_map:
            normal_map_input = normal_batched_renderings[i:i+batch_size]
        depth_map = depth[i:i+batch_size].permute(0, 3, 1, 2).to(device)

        if prompts_list is not None:
            prompt = random.choice(prompts_list)

        with torch.no_grad():
            diffusion_output = run_diffusion_batched(
                                    pipe,
                                    diffusion_input_img,
                                    depth_map,
                                    prompt,
                                    normal_map_input=normal_map_input,
                                    use_latent=use_latent,
                                    num_images_per_prompt=num_images_per_prompt,
                                    return_image=return_image
                                )

        if debug:
            diffusion_time = time()
            t2 = (diffusion_time - predifftime) / 60
            print("Diffusion feature time in mins: ", t2)

        transform = tfs.Compose(
            [
                tfs.Resize((518, 518)),
                tfs.ToTensor(),
                tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        with torch.no_grad():
            aligned_dino_features = [transform(diffusion_output[1][i]) for i in range(batch_size)]
            aligned_dino_features = torch.stack(aligned_dino_features).to(device)
            dheight = aligned_dino_features.shape[2]
            dwidth = aligned_dino_features.shape[3]
            aligned_dino_features = dino_model.get_intermediate_layers(aligned_dino_features, n=1)[0]
            aligned_dino_features = aligned_dino_features.half()

            h, w = int(dheight / patch_size), int(dwidth / patch_size)
            dim = aligned_dino_features.shape[-1]
            aligned_dino_features = aligned_dino_features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
            aligned_dino_features = torch.nn.functional.grid_sample(
                aligned_dino_features, grid[i:i+batch_size], align_corners=False
            ).permute(0, 2, 3, 1) # B x H x W x C

        if normalize:
            aligned_dino_features = torch.nn.functional.normalize(aligned_dino_features, dim=-1)

        if debug:
            dino_time = time()
            t2 = (dino_time - diffusion_time) / 60
            print("DINO feature time in mins: ", t2)

        with torch.no_grad():
            diffusion_output = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0])
            ft_dim = diffusion_output.size(1)
            diffusion_output = torch.nn.functional.grid_sample(
                diffusion_output, grid[i:i+batch_size], align_corners=False
            ).permute(0, 2, 3, 1)

            if normalize:
                diffusion_output = torch.nn.functional.normalize(diffusion_output, dim=-1)

        if debug:
            grid_time = time()
            t2 = (grid_time - dino_time) / 60
            print("Grid upsample time in mins: ", t2)

        # NOTE: Normalization original diff3f features have 1/sqrt(2) norm
        aligned_features = torch.cat([diffusion_output, aligned_dino_features], dim=-1).cpu()

        if normalize:
            aligned_features *= 1/np.sqrt(2) # This ensures unit norm

        tot_aligned_features[i:i+batch_size] = aligned_features

        # tot_aligned_features.append(aligned_features.contiguous())
        if debug:
            cat_time = time()
            t2 = (cat_time - grid_time) / 60
            print("Cat time in mins: ", t2)

    t2 = time() - t1
    t2 = t2 / 60
    print("get_pixel_features_diff3f: Total time taken in mins: ", t2)
    return tot_aligned_features

@torch.no_grad
def get_pixel_features_radio(device, radio_model, image_processor, imgs, H, W, normalize=True,
                              batch_size=10, half=True, debug=False, resize=None,
                              compute_variance=False, compute_cosine=False, patchmask=None):
    model = radio_model

    do_resize = resize is not None
    # NOTE: must be batched with values 0 - 255
    imgs = (imgs * 255).int()
    imgs = imgs[..., :3]
    pixel_values = image_processor(images=imgs, size=resize, return_tensors='pt', do_resize=do_resize,
                                   input_data_format="channels_last").pixel_values
    pixel_values = pixel_values.to(device)

    from tqdm import tqdm

    patch_size = patch_sizes['radio']
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    grid = grid.repeat(len(imgs), 1, 1, 1)

    ### Batch all the renders together ###
    # imgs = imgs[..., :3].permute(0, 3, 1, 2)

    print(f"Getting RADIO features for {imgs.shape} images ...")

    import time
    t0 = time.time()

    features = []
    variances = []
    cosines = []
    for i in tqdm(range(0, len(imgs), batch_size)):
        with torch.no_grad():
            summary, batch_features = model(pixel_values[i:i + batch_size])

            h, w = int(imgs.shape[1] / patch_size), int(imgs.shape[2] / patch_size)
            dim = batch_features.shape[-1]
            batch_features = batch_features.reshape(-1, h, w, dim)

            # NOTE: Features are NOT normalized
            if normalize:
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

            if half:
                batch_features = batch_features.half()
            else:
                batch_features = batch_features.float()

            # Choose LR vs TD vs None for the pixel wiping based on threshold for gap (high gap = strong need for pixel mask)
            # Pixel wiping: use both depths & occupancies (choose starting side for the wipe and ending side determines the baseline depth)
            if compute_cosine and patchmask is not None:
                batch_distances = neighbor_cosine(batch_features, patchmask[i:i+batch_size].to(device))
                cosines.append(batch_distances)

            if compute_variance and patchmask is not None:
                batch_variance = neighbor_variance_8ring(batch_features, patchmask[i:i+batch_size].to(device))

                # Assign variance to patches
                batch_variance = expand_patches(batch_variance, patch_size=patch_size)

                # variance = torch.nn.functional.interpolate(
                #     variance.unsqueeze(1),
                #     size=(H, W),
                #     mode='bilinear',
                #     antialias=True,
                # )
                variances.append(batch_variance)

            # Upsample features to original image size
            batch_features = expand_patches(batch_features, patch_size=patch_size)

        features.append(batch_features.cpu())

    if debug:
        print(f"Total time taken for RADIO features: {time.time() - t0:.2f} seconds")

    if compute_variance and compute_cosine and patchmask is not None:
        cosines = torch.cat(cosines, dim=0).cpu()
        variances = torch.cat(variances, dim=0).cpu()

        return features, cosines, variances

    elif compute_variance and patchmask is not None:
        variances = torch.cat(variances, dim=0).cpu()
        return features, variances

    elif compute_cosine and patchmask is not None:
        cosines = torch.cat(cosines, dim=0).cpu()

        return features, cosines

    else:
        return features

@torch.no_grad
def get_pixel_features_dino(device, dino_model, imgs, H, W, normalize=True,
                              batch_size=10, half=True, debug=False):
    from tqdm import tqdm
    from torchvision import transforms as tfs

    transform = tfs.Compose(
        [
            # NOTE: This is the highest resolution it was finetuned at. dinov2 does not do well at higher res.
            # tfs.Resize((518, 518)),
            # tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    patch_size = patch_sizes['dino2']
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    grid = grid.repeat(len(imgs), 1, 1, 1)

    ### Batch all the renders together ###
    # features = torch.zeros(
    #         (len(imgs), H, W, 1536), device="cpu",
    #         dtype=torch.float16 if half else torch.float32,
    #         )

    imgs = imgs[..., :3].permute(0, 3, 1, 2)

    print(f"Getting DINO features for {imgs.shape} images ...")

    import time
    t0 = time.time()

    features = []
    for i in tqdm(range(0, len(imgs), batch_size)):
        with torch.no_grad():
            batch_imgs = transform(imgs[i:i + batch_size])
            # NOTE: This is the same as running forward_features() and extracting the patch tokens!!
            batch_features = dino_model.get_intermediate_layers(batch_imgs, n=1)[0]

            if half:
                batch_features = batch_features.half()
            else:
                batch_features = batch_features.float()

            h, w = int(batch_imgs.shape[2] / patch_size), int(batch_imgs.shape[3] / patch_size)
            dim = batch_features.shape[-1]
            batch_features = batch_features.reshape(-1, h, w, dim)

            if normalize:
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

            # Upsample features to original image size
            batch_features = expand_patches(batch_features.cpu(), patch_size=patch_size)

        features.append(batch_features)

    if debug:
        print(f"Total time taken for DINO features: {time.time() - t0:.2f} seconds")

    return features

@torch.no_grad
def get_pixel_features_dino3(device, dino_model, imgs, H, W, normalize=True,
                              batch_size=10, half=True, debug=False, resize=None,
                              compute_variance=False, compute_cosine=False,
                              patchmask=None):
    from tqdm import tqdm
    from torchvision import transforms as tfs

    # NOTE: dinov3 should be stable at high resolutions without resizing
    transform = [tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if resize is not None:
        transform.append(tfs.Resize(resize))
    transform = tfs.Compose(transform)

    patch_size = patch_sizes['dino3']
    # grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    # grid = grid.repeat(len(imgs), 1, 1, 1)

    ### Batch all the renders together ###
    imgs = imgs[..., :3].permute(0, 3, 1, 2)

    print(f"Getting DINO v3 features for {imgs.shape} images ...")

    import time
    t0 = time.time()

    features = []
    variances = []
    cosines = []
    for i in tqdm(range(0, len(imgs), batch_size)):
        with torch.no_grad():
            batch_imgs = transform(imgs[i:i + batch_size])
            # NOTE: This is the same as running forward_features() and extracting the patch tokens!!
            batch_features = dino_model.get_intermediate_layers(batch_imgs, n=1)[0]

            h, w = int(batch_imgs.shape[2] / patch_size), int(batch_imgs.shape[3] / patch_size)
            dim = batch_features.shape[-1]
            batch_features = batch_features.reshape(-1, h, w, dim)

            if normalize:
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

            if half:
                batch_features = batch_features.half()
            else:
                batch_features = batch_features.float()

            if compute_cosine and patchmask is not None:
                batch_distances = neighbor_cosine(batch_features, patchmask[i:i+batch_size])
                cosines.append(batch_distances)

            if compute_variance and patchmask is not None:
                variance = neighbor_variance_8ring(batch_features, patchmask[i:i+batch_size])

                # Upsample the variance to the original image size
                patchmask = torch.nn.functional.interpolate(
                    variance,
                    size=(H, W),
                    mode='bilinear',
                    antialias=True,
                )
                variances.append(variance)

            # Upsample features to original image size
            batch_features = expand_patches(batch_features.cpu(), patch_size=patch_size)

        features.append(batch_features)

    if debug:
        print(f"Total time taken for DINO v3 features: {time.time() - t0:.2f} seconds")

    # features = torch.cat(features, dim=0)
    if compute_variance and compute_cosine and patchmask is not None:
        cosines = torch.cat(cosines, dim=0).cpu()
        variances = torch.cat(variances, dim=0).cpu()

        return features, cosines, variances

    elif compute_variance and patchmask is not None:
        variances = torch.cat(variances, dim=0).cpu()
        return features, variances

    elif compute_cosine and patchmask is not None:
        cosines = torch.cat(cosines, dim=0).cpu()

        return features, cosines

    else:
        return features

@torch.no_grad
def get_pixel_features_sam(device, sam_model, imgs, normalize=True,
                            batch_size=5, half=True, debug=False,):
    target_length = sam_model.model.image_encoder.img_size

    imgs = imgs[..., :3].permute(0, 3, 1, 2) # [B, C, H, W]
    oldh, oldw = imgs.shape[2], imgs.shape[3]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    target_size = (newh, neww)

    processed_renderings = torch.nn.functional.interpolate(imgs, target_size, mode="bilinear", antialias=True)
    processed_renderings = processed_renderings.contiguous()
    processed_renderings = sam_model.model.preprocess(processed_renderings)

    from tqdm import tqdm
    featuredim = 256
    view_features = torch.zeros((processed_renderings.shape[0], oldh, oldw, featuredim), device='cpu',
                                dtype=torch.float16 if half else torch.float32)

    print(f"Getting SAM features for {processed_renderings.shape} images ...")
    t0 = time()
    for i in tqdm(range(0, processed_renderings.shape[0], batch_size)):
        with torch.no_grad():
            batch_features = sam_model.model.image_encoder(processed_renderings[i:i+batch_size])

            if half:
                batch_features = batch_features.half()
            else:
                batch_features = batch_features.float()

            # Upsample
            batch_features = torch.nn.functional.interpolate(batch_features, (oldh, oldw), mode="bilinear", antialias=True)

            if normalize:
                batch_features = torch.nn.functional.normalize(batch_features, dim=1)

        view_features[i:i+batch_size] = batch_features.permute(0, 2, 3, 1).cpu()

    if debug:
        print(f"Total time taken for SAM features: {time() - t0:.2f} seconds")

    return view_features

@torch.no_grad
def get_pixel_features_clip(device, clip_model, imgs, normalize=True,
                            clip_conv_layer_weights = [0,0,1.,1.,0],
                            batch_size=20, half=True, debug=False,):
    imgs = imgs[..., :3].permute(0, 3, 1, 2)

    from tqdm import tqdm

    print(f"Getting CLIP features for {imgs.shape} images ...")
    ndim = 1024 # Patch feature dimension
    npatches = 16 # VIT CLIP always resizes to 224x224 with patch size 14

    view_features = torch.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3], ndim), device='cpu',
                            dtype=torch.float16 if half else torch.float32)

    t0 = time()
    for i in tqdm(range(0, imgs.shape[0], batch_size)):
        with torch.no_grad():
            batch_fc_features, batch_conv_features = clip_model(imgs[i:i+batch_size])

            # Aggregate the features
            # NOTE: FC features don't matter
            batch_features = torch.zeros((batch_fc_features.shape[0], ndim, imgs.shape[2], imgs.shape[3]),
                                         dtype=torch.float16 if half else torch.float32, device=device)
            for j, weight in enumerate(clip_conv_layer_weights):
                if weight > 0:
                    # NOTE: first feature is the CLS token
                    batch_conv_feature = batch_conv_features[j][:, 1:, :].reshape(batch_fc_features.shape[0], npatches, npatches, ndim)
                    batch_conv_feature = torch.nn.functional.interpolate(batch_conv_feature.permute(0, 3, 1, 2), (imgs.shape[2], imgs.shape[3]),
                                                                         mode="bilinear", antialias=True)
                    batch_features += batch_conv_feature * weight

            # Upsample
            if normalize:
                batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

        view_features[i:i+batch_size] = batch_features.permute(0, 2, 3, 1).cpu()

    if debug:
        print(f"Total time taken for CLIP features: {time() - t0:.2f} seconds")

    return view_features

@torch.no_grad
def get_pixel_features_sam2(device, sam2_model, imgs, normalize=True,
                            batch_size=20, half=True, debug=False,
                            concat_hr=False,):
    import os
    from tqdm import tqdm

    np_renderings = (imgs.cpu().numpy() * 255).astype(np.uint8)[..., :3]
    np_renderings = [np_renderings[i] for i in range(np_renderings.shape[0])]

    batch_size = 15
    ndim = 256
    if concat_hr:
        ndim = 256 + 32*2
    view_features = torch.zeros((len(imgs), imgs.shape[1], imgs.shape[2], ndim), device='cpu',
                    dtype=torch.float16 if half else torch.float32)
    # TODO: try different options for accumulating the features
    t0 = time()
    for i in tqdm(range(0, len(np_renderings), batch_size)):
        batch = np_renderings[i:i+batch_size]

        # NOTE: This is a hack to get the model to work
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2_model.set_image_batch(batch)

        image_embed = sam2_model._features['image_embed']
        high_res_feats = sam2_model._features['high_res_feats']
        image_embed = torch.nn.functional.interpolate(image_embed, (batch[0].shape[0], batch[0].shape[1]),
                                                                    mode="bilinear", antialias=True)

        if concat_hr:
            hr0 = torch.nn.functional.interpolate(high_res_feats[0], (batch[0].shape[0], batch[0].shape[1]),
                                                                    mode="bilinear", antialias=True)
            hr1 = torch.nn.functional.interpolate(high_res_feats[0], (batch[0].shape[0], batch[0].shape[1]),
                                                                    mode="bilinear", antialias=True)
            image_embed = torch.cat([image_embed, hr0, hr1], dim=1)

        if normalize:
            image_embed = torch.nn.functional.normalize(image_embed, dim=-1)

        if half:
            image_embed = image_embed.half()

        view_features[i:i+batch_size] = image_embed.permute(0, 2, 3, 1).cpu()

    if debug:
        print(f"Total time taken for SAM2 features: {time() - t0:.2f} seconds")

    return view_features