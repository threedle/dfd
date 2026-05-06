"""
Extracted SAM mask generation and feature refinement utilities.
"""
import torch
import numpy as np
import os


def generate_sam_masks(batched_renderings, H, W, patch_size, device,
                       debug=False, debug_dir=None):
    """
    Generate SAM2 segmentation masks for all rendered views.

    Args:
        batched_renderings: (B, H, W, 4) tensor of rendered images
        H, W: image dimensions
        patch_size: model's patch size (used to determine points_per_side)
        device: torch device
        debug: whether to save mask visualizations
        debug_dir: directory for debug output

    Returns:
        batched_sam: list of length B, each element is a list of mask dicts
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam_segmentation import extra_merging
    from GLOBALS import SAM2_CHECKPOINT

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam = build_sam2(
        model_cfg,
        SAM2_CHECKPOINT,
        device=device,
        apply_postprocessing=False
    )

    points_per_side = H // patch_size
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        use_m2m=True,
        box_nms_thresh=0.7,
    )

    batched_sam = []
    render_count = 0
    for render in batched_renderings:
        masks = mask_generator.generate(
            (render.detach().cpu().numpy() * 255).astype(np.uint8)[..., :3]
        )
        masks = extra_merging(masks)
        batched_sam.append(masks)

        if debug and debug_dir is not None:
            _save_sam_debug(render, masks, render_count, debug_dir)
            render_count += 1

    del sam
    del mask_generator

    return batched_sam


def apply_sam_feature_refinement(view_features, batched_sam, pixel_mask,
                                  threshold, featurebatchsize,
                                  H=512, W=512, debug=False, debug_dir=None):
    """
    SAM-segment-driven sub-patch boundary sharpening on patch-resolution
    feature chunks.

    The intent of this refinement is *sub-patch* boundary sharpening: every
    pixel that survives patch-replication upsampling otherwise carries the
    same feature as its containing patch, which blurs over semantic
    boundaries passing through the *interior* of a patch. SAM segments
    capture those boundaries from the original render; replacing outlier
    pixels within each segment with the segment's median feature carves
    feature-patch boundaries along the SAM segmentation, which explicitly
    requires per-pixel granularity.

    Because OPT (5) made ``view_features`` patch-resolution
    (``(A_i, h_patch, w_patch, C)``) we no longer have a full-resolution
    tensor lying around to mutate in place. This function therefore walks
    the input list and replaces each chunk with its patch-replicated
    full-resolution version (``(A_i, H, W, C)``, via ``repeat_interleave``),
    runs the original pixel-level outlier-replacement loop on each view in
    the chunk, and returns the same list now holding the SAM-corrected
    full-resolution chunks. The caller is then expected to mask with
    ``pixel_mask`` to obtain the flattened per-pixel features used downstream
    (the existing gather/flatten code in ``distillation.py`` handles full-
    and patch-resolution chunks uniformly via ``ph = H // h_patch``).

    Args:
        view_features: list of *patch-resolution* feature chunks, each of
            shape ``(A_i, h_patch, w_patch, C)`` on CPU. Modified in place
            (each entry is replaced with its full-resolution equivalent).
        batched_sam: output of ``generate_sam_masks`` (list[list[mask_dict]]).
        pixel_mask: ``(B, H, W)`` boolean mask of valid pixels (CPU).
        threshold: distance threshold for outlier detection
            (``args.use_sam`` value).
        featurebatchsize: number of views per ``view_features`` chunk
            (kept for backward compatibility; the function recovers the
            per-view bookkeeping from each chunk's leading dim directly).
        H, W: original render dimensions.
        debug: whether to save before/after PCA visualizations per view.
        debug_dir: directory for debug output.

    Returns:
        The same ``view_features`` list, now holding full-resolution
        ``(A_i, H, W, C)`` chunks with SAM corrections applied.
    """
    h_patch = view_features[0].shape[1]
    w_patch = view_features[0].shape[2]
    if H % h_patch != 0 or W % w_patch != 0:
        raise ValueError(
            f"Patch grid {h_patch}x{w_patch} does not evenly tile {H}x{W}; "
            f"render H/W must be a multiple of the model's patch size."
        )
    ph = H // h_patch
    pw = W // w_patch

    bi_global = 0
    for ci in range(len(view_features)):
        patch_chunk = view_features[ci]
        chunk_size = patch_chunk.shape[0]

        # Upsample once per chunk: ``repeat_interleave`` allocates a new
        # contiguous tensor (``(A, H, W, C)``) so we can safely mutate slices
        # of it via ``chunk_full[batch_bi]`` views without touching the
        # patch-resolution input. We immediately drop the patch chunk by
        # reassigning the list slot, keeping peak memory bounded to the
        # full-resolution chunks accumulated so far.
        chunk_full = patch_chunk.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)
        view_features[ci] = chunk_full
        del patch_chunk

        for batch_bi in range(chunk_size):
            bi = bi_global + batch_bi
            sam_mask = batched_sam[bi]
            occ_mask_cpu = pixel_mask[bi].squeeze().cpu()
            occ_mask_np = occ_mask_cpu.numpy()

            view_pix = chunk_full[batch_bi]  # writable view into chunk_full

            if debug and debug_dir is not None:
                og_pixel_features = view_pix[occ_mask_cpu].clone()

            for cluster_mask in sam_mask:
                seg = cluster_mask['segmentation']

                # Only include occupied pixels in the cluster
                cluster_pixels = np.where(seg & occ_mask_np)

                # Skip background segmentation (no overlap with mesh)
                if len(cluster_pixels[0]) == 0:
                    continue

                # Skip silhouette segmentation (>90% coverage of union)
                union_sum = np.sum(seg | occ_mask_np)
                if len(cluster_pixels[0]) / union_sum >= 0.9:
                    continue

                # Skip if >50% unoccupied
                occupancy = torch.mean(occ_mask_cpu[seg].float())
                if occupancy < 0.5:
                    continue

                cluster_pixels = (
                    torch.from_numpy(cluster_pixels[0]),
                    torch.from_numpy(cluster_pixels[1]),
                )

                pixel_features = view_pix[cluster_pixels]

                # Compute median feature as cluster representative
                cluster_feature = torch.median(pixel_features, dim=0)[0]
                cluster_feature /= torch.linalg.norm(cluster_feature)

                # Replace outlier pixels with cluster median; subsequent
                # clusters see the modified values, matching the original
                # in-place algorithm.
                distances = torch.linalg.norm(
                    pixel_features - cluster_feature[None], dim=1
                )
                outlier = distances > threshold
                if outlier.any():
                    h_out = cluster_pixels[0][outlier]
                    w_out = cluster_pixels[1][outlier]
                    view_pix[h_out, w_out] = cluster_feature.to(view_pix.dtype)

            if debug and debug_dir is not None:
                _save_refinement_debug(
                    view_pix, og_pixel_features, pixel_mask[bi],
                    bi, H, W, debug_dir,
                )
                del og_pixel_features

        bi_global += chunk_size

    print("Done with feature wiping")
    return view_features


def _save_sam_debug(render, masks, render_count, debug_dir):
    """Save SAM mask visualization for debugging."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    from pathlib import Path

    debugdir = os.path.join(debug_dir, "sam_masks")
    Path(debugdir).mkdir(exist_ok=True, parents=True)

    np.random.seed(3)

    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                        sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                import cv2
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                contours = [
                    cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours
                ]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        ax.imshow(img)

    dpi = 300
    figsize = render.shape[0] / dpi, render.shape[1] / dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.imshow((render.detach().cpu().numpy() * 255).astype(np.uint8)[..., :3])
    show_anns(masks)
    plt.axis('off')
    plt.savefig(os.path.join(debugdir, f"{render_count}.png"))
    plt.close()


def _save_refinement_debug(corrected_features, og_features, mask_bi,
                            bi, H, W, debug_dir):
    """Save before/after PCA visualization of feature refinement."""
    from pathlib import Path
    from sklearn.decomposition import PCA
    from PIL import Image

    debugdir = os.path.join(debug_dir, "sam_debug")
    Path(debugdir).mkdir(exist_ok=True, parents=True)

    pca = PCA(n_components=3)
    mask_np = mask_bi.squeeze().cpu().numpy()
    corrected_np = corrected_features[mask_bi.squeeze().cpu()].detach().cpu().numpy()
    tot_features = np.concatenate([og_features.detach().cpu().numpy(), corrected_np], axis=0)

    features_pca = pca.fit_transform(tot_features)
    features_pca = (features_pca - features_pca.min(axis=0)) / (
        features_pca.max(axis=0) - features_pca.min(axis=0)
    )

    # Original features
    tmp = np.zeros((H, W, 3), dtype=np.uint8)
    tmp[mask_np] = (features_pca[:len(og_features)] * 255).astype(np.uint8)
    Image.fromarray(tmp).save(os.path.join(debugdir, f"{bi}_og.png"))

    # Corrected features
    tmp = np.zeros((H, W, 3), dtype=np.uint8)
    tmp[mask_np] = (features_pca[len(og_features):] * 255).astype(np.uint8)
    Image.fromarray(tmp).save(os.path.join(debugdir, f"{bi}_corrected.png"))
