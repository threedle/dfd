## General image feature distillation into MLP with barycentric-based sampling/coordinates ##
import torch
import argparse
import os
import numpy as np
import time
import diffusers
import gc
from utils import clear_directory, cube_normalize

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def load_mesh(
    meshpath: str,
    *,
    texturedir: str | None = None,
):
    """
    Load a mesh from disk and return tensors needed by the pipeline.

    Supports ``.obj``, ``.ply``, ``.off``, and ``.glb``. For non-GLB formats, vertices are
    cube-normalized after loading. For GLB, vertices are cube-normalized in-place on the
    loaded geometry before tensor conversion.

    Parameters
    ----------
    meshpath:
        Path to the mesh file.
    texturedir:
        Optional texture image path used for textured OBJ rendering (UVs).

    Returns
    -------
    glbmesh:
        Trimesh object of the mesh. Only returned for .glb files.
    mesh_vertices:
        Float tensor of shape ``(V, 3)``.
    mesh_faces:
        Long tensor of shape ``(F, 3)``.
    vertex_colors:
        Optional float tensor of shape ``(V, 3)`` in ``[0, 1]`` if available, otherwise a default gray.
    vertex_normals:
        Currently unused (always ``None``).
    texture_data:
        Optional dict containing UV coordinates/faces and texture image for OBJ meshes.
    """
    # Use trimesh/igl for all formats — no PyTorch3D dependency
    vertex_colors = None
    vertex_normals = None
    texture_data = None
    glbmesh = None

    if meshpath.endswith(".obj"):
        import igl

        v, vt, n, f, ftc, _ = igl.readOBJ(meshpath)
        verts = torch.from_numpy(cube_normalize(v[:, :3])).float()
        faces = torch.from_numpy(f).long()

        # Check for meshlab vertex color formatting (6-column vertices)
        if v.shape[-1] == 6:
            vertex_colors = torch.from_numpy(v[:, 3:]).float()
        else:
            vertex_colors = torch.ones_like(verts) * 0.8

        # Handle UV textures
        if texturedir is not None and vt is not None and len(vt) > 0:
            import torchvision

            textureimg = torchvision.io.read_image(texturedir).permute(1, 2, 0).float() / 255.0
            # Flip for nvdiffrast
            textureimg = torch.flip(textureimg, dims=(0,))
            uv_coords = torch.from_numpy(vt).float()
            uv_faces = torch.from_numpy(ftc).long() if ftc is not None and len(ftc) > 0 else faces
            texture_data = {
                "uv_coords": uv_coords,
                "uv_faces": uv_faces,
                "texture_image": textureimg.unsqueeze(0),  # (1, H, W, 3)
            }

    elif meshpath.endswith(".ply"):
        import trimesh

        tm = trimesh.load(meshpath)
        verts = torch.from_numpy(cube_normalize(np.array(tm.vertices))).float()
        faces = torch.from_numpy(np.array(tm.faces)).long()
        vertex_colors = torch.ones_like(verts) * 0.8

    elif meshpath.endswith(".off"):
        import igl

        v, f = igl.read_triangle_mesh(meshpath)
        verts = torch.from_numpy(cube_normalize(v)).float()
        faces = torch.from_numpy(f).long()
        vertex_colors = torch.ones_like(verts) * 0.8

    elif meshpath.endswith(".glb"):
        import trimesh

        glbmesh = trimesh.load(meshpath)
        if isinstance(glbmesh, trimesh.Scene):
            glbmesh = glbmesh.to_geometry()

        glbmesh.vertices = cube_normalize(glbmesh.vertices)
        verts = torch.from_numpy(np.array(glbmesh.vertices)).float()
        faces = torch.from_numpy(np.array(glbmesh.faces)).long()
    else:
        raise NotImplementedError("Only .obj, .ply, .off, and .glb files are supported")

    mesh_vertices = verts.float()
    mesh_faces = faces.long()
    return glbmesh, mesh_vertices, mesh_faces, vertex_colors, vertex_normals, texture_data


def barycentric_distillation(
    *,
    meshpath: str,
    savedir: str,
    texturedir: str | None = None,
    viewradius: float = 1,
    overwrite: bool = False,
    debug: bool = False,
    timing: bool = False,
    no_cache: bool = False,
    reduction: float = 0,
    model: str = "dino2",
    arch: str = None,
    checkpoint: str = None,
    repodir: str = None,
    model_cfg: str = None,
    sam2_hr: bool = False,
    imgh: int = 512,
    imgw: int = 512,
    nviews: int = 16,
    viewtype: str = "default",
    batchsize: int = 10000,
    viewbatchsize: int = 16,
    featurebatchsize: int = 2,
    lr: float = 1e-3,
    iters: int = 1000,
    noiseradius: float = 0.05,
    noisen: int = 0,
    subsetepoch: float = 0,
    use_sam: float = 0,
    nlayers: int = 4,
    width: int = 256,
    positional_encoding: bool = False,
    sigma: float = 5.0,
    normalizemlp: bool = True,
    saveto: str = 'vertices', # vertices or faces
):
    """
    Distill per-pixel image features into feature field via barycentric sampling.

    This function runs the full distillation pipeline:

    - Load and normalize a mesh (supports ``.obj``, ``.ply``, ``.off``, ``.glb``).
    - Render the mesh from multiple views and compute per-pixel 3D positions (barycentric
      coordinates/triangle correspondence) plus a visibility mask.
    - Extract dense per-pixel features with the selected image model (e.g. DINO/CLIP/SAM/RADIO).
    - Fit an MLP that maps 3D positions to the corresponding image feature vectors.
    - Predict a feature for each face using triangle centroids and save the result.
    - Optionally produce PCA-colored screenshots using Polyscope.

    Parameters
    ----------
    meshpath:
        Path to the input mesh.
    savedir:
        Output directory. The function creates it if missing and writes checkpoints and outputs here.
    texturedir:
        Optional texture image path used for textured OBJ rendering (UVs).
    viewradius, nviews, viewtype, viewbatchsize, imgh, imgw:
        Rendering configuration.
    model, sam2_hr:
        Feature extractor selection and model-specific flags.
    overwrite:
        If True, clears ``savedir`` before running.
    reduction:
        If > 0, simplifies the input mesh before processing (percentage of edges to collapse).
        The simplified mesh path replaces ``meshpath`` for the remainder of the run.
    debug, time:
        Debug dumps and timing prints.
    no_cache:
        If True, avoids writing cached intermediates to disk (saves disk space).
    batchsize:
        Number of (flattened) training points sampled per optimization step.
    lr, iters, subsetepoch:
        Training hyperparameters.
    noiseradius, noisen:
        If ``noisen > 0``, samples additional noisy 3D points around each position during training.
    use_sam:
        If > 0, uses SAM masks to refine per-pixel features and reduce feature bleeding.
        The value is used as the refinement threshold.
    nlayers, width, positional_encoding, sigma, normalizemlp:
        MLP architecture and encoding options.

    Outputs
    -------
    Returns: optimized MLP feature field and predicted features on either vertices or faces depending on saveto.

    Also saves the following files to disk:
    - ``{savedir}/{meshname}.pt``: distilled features (tensor) on either vertices or faces depending on saveto.
    - ``{savedir}/final_encoder.pth`` (+ ``latest_encoder.pth``, ``latest_iter``, ``loss_list.npy``):
      trained MLP checkpoints.
    - ``{savedir}/loss.png``: training loss curve.
    - ``{savedir}/cache/``: cached render,depth,normal buffers / pixel coordinates / masks (unless ``no_cache``).
    - ``{savedir}/cache/pca*.png``: PCA visualization screenshots.
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    from pathlib import Path
    Path(savedir).mkdir(parents=True, exist_ok=True)

    if overwrite:
        clear_directory(savedir)

    if os.path.exists(os.path.join(savedir, "final_encoder.pth")):
        print("Final encoder already exists. Skipping.")
        return None, None

    cachedir = os.path.join(savedir, "cache")
    Path(cachedir).mkdir(parents=True, exist_ok=True)

    if reduction > 0:
        if timing:
            t0 = time.time()


        if meshpath.endswith(".glb"):
            from simplify_mesh import simplify_from_gltf_file
            savepath = os.path.join(cachedir, os.path.basename(meshpath).split(".")[0] + f"_simplified{reduction}.glb")
            simplify_from_gltf_file(meshpath, reduction, savepath)
            meshpath = savepath
        else:
            from simplify_mesh import simplify
            meshname = os.path.basename(meshpath).split(".")[0]
            savepath = os.path.join(cachedir, f"{meshname}_simplified{reduction}.obj")

            if os.path.exists(savepath):
                print(f"Simplified mesh already exists at {savepath}")
                meshpath = savepath
            else:
                import igl
                if meshpath.endswith(".obj"):
                    v, vt, n, f, ftc, _ = igl.readOBJ(meshpath)
                else:
                    v, f = igl.read_triangle_mesh(meshpath)
                    vt = None
                    ftc = None
                simplify(v, f, reduction, savepath, uvs=vt, fuv=ftc)
            meshpath = savepath

        if timing:
            simplifytime = time.time() - t0
            with open(os.path.join(savedir, "simplifytime.txt"), "w") as f:
                f.write(f"{simplifytime:.8f}\n")

    # No warnings!!
    diffusers.logging.set_verbosity_error()

    meshname = os.path.basename(meshpath).split(".")[0]

    # ==================== Mesh Loading ====================
    glbmesh, mesh_vertices, mesh_faces, vertex_colors, vertex_normals, texture_data = load_mesh(
        meshpath,
        texturedir=texturedir,
    )

    # ==================== MLP Training Setup ====================
    H = imgh
    W = imgw
    use_normal_map = False
    t0 = time.time()

    if not os.path.exists(os.path.join(savedir, f"{meshname}.pt")) or not os.path.exists(os.path.join(savedir, f"final_encoder.pth")):

        # ==================== Rendering ====================
        if not os.path.exists(os.path.join(cachedir, f"renders.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"pixel_coords.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"depth.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"pixel_mask.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"xyz.pt")):

            prerendertime = time.time()

            if meshpath.endswith(".glb"):
                from render_pyrender import run_rendering

                normal_batched_renderings = None
                batched_renderings, depth, pixel_mask, pixel_coords = run_rendering(
                    device, glbmesh, nviews, H, W,
                    radius=viewradius, viewtype=viewtype,
                )
            else:
                from render_nvdiffrast import run_rendering
                import nvdiffrast.torch as dr
                glctx = dr.RasterizeCudaContext()

                batched_renderings, normal_batched_renderings, depth, pixel_mask, pixel_coords = run_rendering(
                    glctx, device, mesh_vertices, mesh_faces, nviews, H, W,
                    vertex_colors=vertex_colors, vertex_normals=vertex_normals,
                    use_normal_map=use_normal_map, radius=viewradius, viewtype=viewtype,
                    texture_data=texture_data, batch_size=viewbatchsize,
                )

            if timing:
                rendertime = time.time() - prerendertime
                print(f"Rendering complete in {rendertime} seconds")

            # Compute the boolean visibility mask once and reuse for: (1) gathering
            # valid xyz, and (2) the (b, h, w) indices used to gather flat features
            # after extraction.
            flat_mask = pixel_mask.bool()
            xyz = pixel_coords[flat_mask].cpu().contiguous()

            if not no_cache:
                torch.save(batched_renderings.cpu(), os.path.join(cachedir, f"renders.pt"))
                torch.save(depth.cpu(), os.path.join(cachedir, f"depth.pt"))
                torch.save(pixel_mask.cpu(), os.path.join(cachedir, f"pixel_mask.pt"))
                torch.save(pixel_coords.cpu(), os.path.join(cachedir, f"pixel_coords.pt"))
                torch.save(xyz, os.path.join(cachedir, f"xyz.pt"))
                if use_normal_map and normal_batched_renderings is not None:
                    torch.save(normal_batched_renderings.cpu(), os.path.join(cachedir, f"normal_renders.pt"))

            # mesh_vertices/faces are already CPU from load_mesh; no work needed.
            # pixel_coords is no longer needed (xyz is the masked subset);
            # depth is only consumed by extractors with needs_normal_map=True.
            pixel_mask = pixel_mask.cpu()
            del pixel_coords, flat_mask
            if use_normal_map:
                depth = depth.cpu()
            else:
                depth = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            batched_renderings = torch.load(os.path.join(cachedir, f"renders.pt"), map_location=device, weights_only=True)
            pixel_mask = torch.load(os.path.join(cachedir, f"pixel_mask.pt"), map_location='cpu', weights_only=True)
            xyz = torch.load(os.path.join(cachedir, f"xyz.pt"), map_location='cpu', weights_only=True)

            # depth is only consumed by extractors that need normal maps.
            depth = None
            if use_normal_map:
                depth = torch.load(os.path.join(cachedir, f"depth.pt"), map_location=device, weights_only=True)

            normal_batched_renderings = None
            if use_normal_map and os.path.exists(os.path.join(cachedir, f"normal_renders.pt")):
                normal_batched_renderings = torch.load(os.path.join(cachedir, f"normal_renders.pt"), map_location=device, weights_only=True)

            print(f"Loaded cached render data from {cachedir}")

        if debug:
            from pathlib import Path
            renderdir = os.path.join(cachedir, "renders")
            Path(renderdir).mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for i in range(batched_renderings.shape[0]):
                img = (batched_renderings[i].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(renderdir, f"render_{i:02}.png"))

        # ==================== SAM Mask Generation ====================
        # Skip SAM mask generation if the view_features cache already exists — SAM
        # refinement is already baked into cached features.
        if use_sam > 0 and not os.path.exists(os.path.join(cachedir, f"view_features.pt")):
            print("Generating segmentation masks using SAM ...")
            from sam_utils import generate_sam_masks
            from GLOBALS import patch_sizes

            if timing:
                presamtime = time.time()

            batched_sam = generate_sam_masks(
                batched_renderings, H, W, patch_sizes[model], device,
                debug=debug, debug_dir=cachedir,
            )

            if timing:
                samtime = time.time() - presamtime
                print(f"SAM segmentation completed in {samtime:02f} seconds")

        # ==================== Feature Extraction ====================
        # Cached features are stored *flat* — i.e. only valid (mask=True) pixels are kept,
        # aligned with the flat ``xyz`` tensor of training positions.
        view_features_path = os.path.join(cachedir, f"view_features.pt")

        if not os.path.exists(view_features_path):
            print("Preprocessing all the renders")
            prefeaturetime = time.time()

            from feature_extractor import create_extractor

            extractor = create_extractor(model, device, arch=arch, checkpoint=checkpoint,
                                         repodir=repodir, model_cfg=model_cfg)

            # Build kwargs for feature extraction
            feat_kwargs = dict(
                batch_size=featurebatchsize,
                normalize=True,
                debug=debug,
            )

            # Model-specific extra kwargs
            if extractor.needs_normal_map:
                feat_kwargs['normal_renderings'] = normal_batched_renderings
                feat_kwargs['depth'] = depth

            if model == "sam2":
                feat_kwargs['concat_hr'] = sam2_hr

            # NOTE: We do not concatenate extract features because memory cost too high
            view_features = extractor.get_pixel_features(
                batched_renderings, H, W, **feat_kwargs
            )
            
            extractor.cleanup()

            # Apply SAM feature refinement if applicable. Because
            # ``view_features`` is patch-resolution, the refinement
            # materializes the full-resolution feature map (via
            # ``repeat_interleave``) and runs the original pixel-level
            # outlier replacement on it. The function returns the
            # *full-resolution* view_features list with SAM corrections
            # baked in, which the gather/flatten code below then masks
            # down to flat per-pixel features (the patch-resolution
            # gather works uniformly for full-res chunks since
            # ``ph = H // h_patch = 1`` when ``h_patch = H``).
            if use_sam > 0 and extractor.supports_sam_refinement:
                from sam_utils import apply_sam_feature_refinement
                sam_view_features = apply_sam_feature_refinement(
                    view_features, batched_sam, pixel_mask,
                    threshold=use_sam,
                    featurebatchsize=view_features[0].shape[0],
                    H=H, W=W,
                    debug=debug,
                    debug_dir=cachedir,
                )

                # Flatten by walking the (now full-resolution) chunks and
                # masking each with its slice of ``pixel_mask``. We
                # pre-allocate the flat output buffer and stream chunk by
                # chunk, freeing each upsampled chunk as soon as it is
                # consumed - so we never hold both the upsampled list and a
                # concatenated copy at the same time.
                #
                # Per-chunk masking preserves global row-major enumeration
                # over ``(B, H, W)``: applying ``pixel_mask[chunk_start:chunk_end]``
                # to each chunk and concatenating the per-chunk row counts is
                # identical to applying the full mask to a hypothetical
                # ``cat()``ed tensor, which matches the order of
                # ``xyz = pixel_coords[pixel_mask.bool()]``.
                pixel_mask_bool = pixel_mask.bool()
                n_valid = int(pixel_mask_bool.sum().item())
                feat_dim = sam_view_features[0].shape[-1]
                flat_features = torch.empty(
                    (n_valid, feat_dim), dtype=sam_view_features[0].dtype
                )

                chunk_start = 0
                flat_off = 0
                for i in range(len(sam_view_features)):
                    chunk = sam_view_features[i]
                    chunk_end = chunk_start + chunk.shape[0]
                    chunk_flat = chunk[pixel_mask_bool[chunk_start:chunk_end]]
                    n_chunk = chunk_flat.shape[0]
                    flat_features[flat_off:flat_off + n_chunk] = chunk_flat
                    flat_off += n_chunk
                    chunk_start = chunk_end
                    sam_view_features[i] = None
                    del chunk, chunk_flat

                assert flat_off == n_valid, (
                    f"Flat fill mismatch: wrote {flat_off} rows but expected {n_valid}"
                )

                del sam_view_features, pixel_mask_bool
                view_features = flat_features
                del flat_features
            else:
                # Drop background/invalid pixels and flatten to [N_valid, C] without ever
                # cat()/stack()ing the (potentially huge) per-chunk feature tensors.
                #
                # ``view_features`` is now stored at *patch resolution*: each chunk has
                # shape ``(A_i, h_patch, w_patch, C)`` rather than ``(A_i, H, W, C)``,
                # which avoids materializing a ~``patch**2`` larger upsampled tensor
                # per chunk (e.g. ~196x for DINOv2 with patch=14). We replicate the
                # per-pixel features on the fly here by integer-dividing the pixel
                # indices by the patch size: pixel ``(h, w)`` reads from patch
                # ``(h // ph, w // pw)``.
                #
                # Order matches ``xyz = pixel_coords[pixel_mask.bool()]``: both
                # ``tensor[bool_mask]`` and ``torch.where(bool_mask)`` enumerate in
                # row-major order over [B, H, W], so the i-th feature row aligns
                # with ``xyz[i]``.
                first_chunk = view_features[0]
                h_patch = first_chunk.shape[1]
                w_patch = first_chunk.shape[2]
                assert H % h_patch == 0 and W % w_patch == 0, (
                    f"Patch grid {h_patch}x{w_patch} does not evenly tile {H}x{W}; "
                    f"render H/W must be a multiple of the model's patch size."
                )
                ph = H // h_patch
                pw = W // w_patch

                b_idx, h_idx, w_idx = torch.where(pixel_mask.bool())
                patch_h_idx = h_idx // ph
                patch_w_idx = w_idx // pw
                del h_idx, w_idx
                n_valid = b_idx.numel()
                feat_dim = first_chunk.shape[-1]
                flat_features = torch.empty(
                    (n_valid, feat_dim), dtype=first_chunk.dtype
                )

                chunk_start = 0
                # Iterate by index so we can release each chunk as soon as it is consumed,
                # keeping CPU peak ~= one chunk + flat_features instead of the entire list.
                for i in range(len(view_features)):
                    chunk = view_features[i]
                    chunk_end = chunk_start + chunk.shape[0]
                    sel = torch.nonzero(
                        (b_idx >= chunk_start) & (b_idx < chunk_end),
                        as_tuple=False,
                    ).squeeze(1)
                    if sel.numel() > 0:
                        local_b = b_idx[sel] - chunk_start
                        flat_features[sel] = chunk[local_b, patch_h_idx[sel], patch_w_idx[sel]]
                    chunk_start = chunk_end
                    view_features[i] = None
                    del chunk

                del view_features, b_idx, patch_h_idx, patch_w_idx, first_chunk
                view_features = flat_features
                del flat_features

            # Clean up rendering data
            del batched_renderings, depth

            if timing:
                featuretime = time.time() - prefeaturetime
                print(f"Feature extraction complete in {featuretime/60} minutes")

            gc.collect()
            torch.cuda.empty_cache()

            if not no_cache:
                torch.save(view_features, view_features_path)
                print(f"Saved flat view features to {view_features_path}")
        else:
            preloadtime = time.time()
            view_features = torch.load(view_features_path, map_location='cpu', weights_only=True)
            print(f"Loaded cached view features from {view_features_path}")
            if debug:
                t2 = (time.time() - preloadtime) / 60
                print(f"Time to load view features: {t2} minutes")

        # pixel_mask is no longer needed once features are flat; keep it on disk only.
        del pixel_mask
        gc.collect()

        assert view_features.shape[0] == xyz.shape[0], (
            f"Flattened feature/xyz counts disagree: "
            f"{view_features.shape[0]} vs {xyz.shape[0]}"
        )
        print(f"Training points (valid pixels): {view_features.shape[0]}, "
              f"feature dim: {view_features.shape[1]}")

        if timing:
            pretraintime = time.time()

        # ==================== MLP Training ====================
        mesh_vertices = mesh_vertices.cpu()
        mesh_faces = mesh_faces.cpu()

        # `view_features` is a flat [N_valid, C] CPU tensor and `xyz` is the aligned
        # [N_valid, 3] CPU tensor of valid training positions. `batchsize` counts
        # flattened training points per optimization step.
        n_total_points = view_features.shape[0]
        if subsetepoch > 0:
            n_iter_points = max(1, int(np.floor(n_total_points * subsetepoch)))
            print(f"[subsetepoch] Sampling {n_iter_points} / {n_total_points} points "
                  f"({subsetepoch*100:.1f}%) per iter, batchsize={batchsize}")
        else:
            n_iter_points = n_total_points

        from featuremlp import FeatureMLP

        ndim = view_features.shape[-1]
        encoder = FeatureMLP(nlayers, width, out_dim=ndim, positional_encoding=positional_encoding,
                             sigma=sigma, normalize=normalizemlp).to(device)

        # Check for latest encoder checkpoint
        startiter = 0
        if os.path.exists(os.path.join(savedir, f"latest_encoder.pth")) and os.path.exists(os.path.join(savedir, f"latest_iter")):
            print(f"Loading encoder from {os.path.join(savedir, 'latest_encoder.pth')}")
            encoder.load_state_dict(torch.load(os.path.join(savedir, f"latest_encoder.pth"), map_location=device))
            startiter = int(open(os.path.join(savedir, f"latest_iter")).readline().strip())
            print(f"Starting from iteration {startiter}")
            loss_list = np.load(os.path.join(savedir, f"loss_list.npy"), allow_pickle=False).tolist()
        else:
            loss_list = []

        optim = torch.optim.Adam(encoder.parameters(), lr=lr)

        from tqdm import tqdm

        for iteri in tqdm(range(startiter, iters)):
            # Random permutation over all valid points each iter; if subsetepoch > 0,
            # take only the first `n_iter_points` of the permutation as this iter's subset.
            perm = torch.randperm(n_total_points)[:n_iter_points]

            iter_features = None
            iter_xyz = None
            
            # One large H2D copy per iter (instead of one per mini-batch).
            iter_features = view_features[perm].to(device, non_blocking=True)
            iter_xyz = xyz[perm].to(device, non_blocking=True)

            totloss_gpu = torch.zeros((), device=device, dtype=torch.float32)
            n_batches = 0

            for start in tqdm(range(0, n_iter_points, batchsize)):
                end = min(start + batchsize, n_iter_points)
                gtfeatures = iter_features[start:end]
                positions = iter_xyz[start:end]

                if noisen > 0:
                    n = len(positions)
                    raw = torch.randn(n, noisen, 5, device=device)
                    norm = raw.norm(dim=-1, keepdim=True) / noiseradius
                    noise = raw[..., 2:] / norm
                    noisy_positions = positions.unsqueeze(1) + noise
                    positions = torch.cat([positions.unsqueeze(1), noisy_positions], dim=1).reshape(-1, 3)
                    gtfeatures = gtfeatures.repeat_interleave(noisen + 1, dim=0)

                pred_features = encoder(positions)
                # Upcast targets to MLP dtype only when needed.
                if gtfeatures.dtype != pred_features.dtype:
                    gtfeatures = gtfeatures.to(pred_features.dtype)
                batch_loss = torch.nn.functional.mse_loss(
                    pred_features, gtfeatures, reduction='sum'
                )

                optim.zero_grad(set_to_none=True)
                batch_loss.backward()
                optim.step()

                # Accumulate on GPU; avoid a per-batch .item() sync.
                totloss_gpu += batch_loss.detach()
                n_batches += 1

            del iter_features, iter_xyz

            totloss = (totloss_gpu / max(1, n_batches)).item()
            print(f"Iter {iteri} loss: {totloss}")
            loss_list.append(totloss)

            if iteri == iters - 1 or (iteri + 1) % 5 == 0:
                np.save(os.path.join(savedir, f"loss_list.npy"), loss_list)
                torch.save(encoder.state_dict(), os.path.join(savedir, f"latest_encoder.pth"))
                with open(os.path.join(savedir, f"latest_iter"), "w") as f:
                    f.write(str(iteri + 1))

        # Save final model
        encoder.eval()
        torch.save(encoder.state_dict(), os.path.join(savedir, f"final_encoder.pth"))
        print(f"Saved encoder to {os.path.join(savedir, 'final_encoder.pth')}")

        import json
        with open(os.path.join(savedir, "final_encoder.json"), 'w') as f:
            jsonargs = vars(args)
            jsonargs['out_dim'] = ndim
            json.dump(vars(args), f)

        if timing:
            traintime = time.time() - pretraintime
            print(f"Time to train MLP: {traintime/60} minutes")

            tottime = rendertime + featuretime + traintime
            if use_sam > 0:
                tottime += samtime

            with open(os.path.join(savedir, f"timing.txt"), "w") as f:
                f.write(f"{rendertime:.8f}\n")
                if use_sam > 0:
                    f.write(f"{samtime:.8f}\n")
                f.write(f"{featuretime:.8f}\n")
                f.write(f"{traintime:.8f}\n")
                f.write(f"{tottime:.8f}\n")

        elapsed = time.time() - t0
        elapsed = elapsed / 60
        print(f"Total time from rendering to fitting MLP: {elapsed} minutes")

        # ==================== Feature Prediction ====================
        if saveto == 'vertices':
            sample_points = mesh_vertices.to(device)
        elif saveto == 'faces':
            sample_points = (mesh_vertices[mesh_faces].mean(dim=1)).to(device)
        else:
            raise ValueError(f"Invalid saveto value: {saveto}")

        with torch.no_grad():
            features = encoder(sample_points)

        torch.save(features.detach().cpu(), os.path.join(savedir, f"{meshname}.pt"))
        print(f"Saved features to {os.path.join(savedir, f'{meshname}.pt')}")

        # Plot the loss
        import matplotlib.pyplot as plt
        plt.plot(np.arange(iters), loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join(savedir, f"loss.png"))
        plt.close()
    else:
        print(f"Loading features from {os.path.join(savedir, f'{meshname}.pt')}")
        features = torch.load(os.path.join(savedir, f"{meshname}.pt"), weights_only=True, map_location=device)

    # ==================== PCA Visualization ====================
    print(f"Saving PCA visualization to {cachedir} ...")
    import polyscope as ps
    from sklearn.decomposition import PCA
    from utils import gen_elev_azim, get_pos_from_elev
    import math

    elev_samples = 1
    azim_samples = 6
    start_elev = math.radians(0)
    end_elev = math.radians(30)
    start_azim = math.radians(0)
    end_azim = math.radians(360 - 360/azim_samples)

    elev, azim = gen_elev_azim(start_elev, end_elev, elev_samples, start_azim, end_azim, azim_samples,
                                device=device)

    positions = get_pos_from_elev(elev, azim, viewradius, blender=False, device=device).detach().cpu().numpy()
    lookats = np.zeros_like(positions)

    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0) - features_pca.min(axis=0))

    ps.set_allow_headless_backends(True)
    ps.init()
    ps.remove_all_structures()

    ps_mesh = ps.register_surface_mesh(
        meshname,
        mesh_vertices.detach().cpu().numpy(),
        mesh_faces.detach().cpu().numpy(),
    )
    ps_mesh.add_color_quantity(
        "features",
        features_pca,
        defined_on=saveto, enabled=True)

    # Set camera fov to 60
    ps.set_vertical_fov_degrees(60)

    for i in range(len(positions)):
        ps.look_at(positions[i], lookats[i])
        ps.screenshot(os.path.join(cachedir, f"pca{i}.png"))

    return encoder, features

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("meshpath", type=str)
    parse.add_argument("savedir", type=str)
    parse.add_argument("--texturedir", type=str, default=None)
    parse.add_argument("--viewradius", type=float, default=2.8)
    parse.add_argument("--saveto", type=str, choices={'vertices', 'faces'}, default='vertices', help="where to sample the features from")
    parse.add_argument("--overwrite", action="store_true")
    parse.add_argument("--debug", action="store_true")
    parse.add_argument("--timing", action="store_true")
    parse.add_argument('--no_cache', action="store_true", help="don't cache anything. saves on disk space.")

    ### Simplification parameters
    parse.add_argument("--reduction", type=float, default=0, help="What percentage of edges to collapse.")

    ### Image model parameters
    parse.add_argument("--model", choices={"dino2", "dino3", 'clip', 'sam', 'sam2', 'radio'}, type=str, default="dino2")
    parse.add_argument("--arch", type=str, default=None, help="specific architecture name for the model")
    parse.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint weights if stored locally")
    parse.add_argument("--repodir", type=str, default=None, help="path to repository source for the model if stored locally")
    parse.add_argument("--model_cfg", type=str, default=None, help="path to model configuration file if stored locally. Only relevant for sam2 currently.")
    parse.add_argument('--sam2_hr', action="store_true")

    ### Training parameters
    parse.add_argument("-H", "--imgh", type=int, default=512)
    parse.add_argument("-W", "--imgw", type=int, default=512)

    parse.add_argument("--nviews", type=int, default=24, help="total views to render (should be multiple of 3 for default viewtype)")
    parse.add_argument("--viewtype", type=str, choices={'default', 'fib'}, default='default', help="default view sampling or fibonacci")
    parse.add_argument("--batchsize", type=int, default=10000, help="number of (flattened) training points sampled per MLP optimization step")
    parse.add_argument("--viewbatchsize", type=int, default=16, help="number of views to batch for rendering")
    parse.add_argument("--featurebatchsize", type=int, default=2, help="number of views to batch for feature extraction")
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--iters", type=int, default=20)

    ### Gaussian blurring
    parse.add_argument("--noiseradius", type=float, default=0.05, help="maximum radius for sampling outside of the vertex")
    parse.add_argument("--noisen", type=int, default=0, help="noise samples for every vertex")

    ### Subset epoch training
    parse.add_argument('--subsetepoch', type=float, default=0.1, help="If > 0, randomly samples this fraction of all flattened training points each iter.")

    ### SAM feature reassignment
    parse.add_argument('--use_sam', type=float, default=0, help="SAM segmentation to fix feature bleeding. Value determines pixel features which are outliers from the cluster mode.")

    ### MLP parameters
    parse.add_argument('--nlayers', type=int, default=4)
    parse.add_argument('--width', type=int, default=256)
    parse.add_argument('--positional_encoding', action="store_true", help="using fourier features for positional encoding")
    parse.add_argument('--sigma', type=float, default=5.0, help="sigma for fourier features")

    args = parse.parse_args()
    barycentric_distillation(**vars(args))
