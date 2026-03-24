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

        # TODO: get uv coordinates and texture image if available
    else:
        raise NotImplementedError("Only .obj, .ply, .off, and .glb files are supported")

    mesh_vertices = verts.float()
    mesh_faces = faces.long()
    return glbmesh, mesh_vertices, mesh_faces, vertex_colors, vertex_normals, texture_data


def barycentric_distillation(
    *,
    meshpath: str,
    savedir: str,
    description=None,
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
    flattenviews: bool = False,
    batchsize: int = 5,
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
    - Extract dense per-pixel features with the selected image model (e.g. Diff3F/DINO/CLIP/SAM/RADIO).
    - Fit an MLP that maps 3D positions to the corresponding image feature vectors.
    - Predict a feature for each face using triangle centroids and save the result.
    - Optionally produce PCA-colored screenshots using Polyscope.

    Parameters
    ----------
    meshpath:
        Path to the input mesh.
    savedir:
        Output directory. The function creates it if missing and writes checkpoints and outputs here.
    description:
        Text prompt for models that need it (e.g. ``diff3f``). This matches the CLI behavior:
        typically a list of tokens from argparse (``nargs="+"``). When empty and ``model=="diff3f"``,
        the mesh basename is used.
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
    flattenviews, batchsize:
        Dataset/training batching behavior.
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

    # Normalize/compat: argparse uses [] for unset description; avoid mutable default in signature.
    if description is None:
        description = []

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    from pathlib import Path
    Path(savedir).mkdir(parents=True, exist_ok=True)

    if overwrite:
        clear_directory(savedir)

    if os.path.exists(os.path.join(savedir, "final_encoder.pth")):
        print("Final encoder already exists. Existing...")
        exit(0)

    cachedir = os.path.join(savedir, "cache")
    Path(cachedir).mkdir(parents=True, exist_ok=True)

    if reduction > 0:
        if timing:
            t0 = time.time()


        if meshpath.endswith(".glb"):
            from simplify_mesh import simplify_from_gltf_file
            savepath = str(os.path.basename(meshpath))[:-3] + f"_simplified{reduction}.glb"
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
    if len(description) > 0:
        description = " ".join(description)
    else:
        description = ""

    if len(description) == 0 and model == "diff3f":
        print(f"No description provided. Using meshname as description for diff3f distillation: {meshname}")
        description = meshname

    # ==================== Mesh Loading ====================
    glbmesh, mesh_vertices, mesh_faces, vertex_colors, vertex_normals, texture_data = load_mesh(
        meshpath,
        texturedir=texturedir,
    )

    # ==================== Dataset class ====================
    from torch.utils.data import Dataset, DataLoader

    class FeatureDataset(Dataset):
        def __init__(self, features, positions, masks=None, flatten=False, featurebatchsize=None):
            """features: B-list of shape [H, W, C]
            positions: [B, H, W, 3]
            masks: [B, H, W], Optional
            flatten: whether to flatten the views
            """
            self.featurebatchsize = featurebatchsize
            self.flatten = flatten
            if self.flatten:
                features = torch.cat(features, dim=0)
                self.features = features.view(-1, features.shape[-1])
                self.positions = positions.view(-1, 3)
                if masks is not None:
                    self.masks = masks.view(-1)
                    self.features = self.features[self.masks]
                    self.positions = self.positions[self.masks]
                else:
                    self.masks = None
            else:
                self.features = features
                self.positions = positions
                self.masks = masks

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, idx):
            if self.flatten:
                if self.masks is not None:
                    return self.features[idx], self.positions[idx], self.masks[idx]
                else:
                    return self.features[idx], self.positions[idx], torch.tensor(True)
            else:
                view_bi = idx // self.featurebatchsize
                batch_bi = idx % self.featurebatchsize
                if self.masks is not None:
                    return self.features[view_bi][batch_bi], self.positions[idx], self.masks[idx]
                else:
                    return self.features[view_bi][batch_bi], self.positions[idx], torch.tensor(True)

    # ==================== MLP Training Setup ====================
    H = imgh
    W = imgw
    use_normal_map = model == "diff3f"

    t0 = time.time()

    if not os.path.exists(os.path.join(savedir, f"{meshname}.pt")) or not os.path.exists(os.path.join(savedir, f"final_encoder.pth")):

        # ==================== Rendering ====================
        if not os.path.exists(os.path.join(cachedir, f"renders.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"pixel_coords.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"depth.pt")) or \
            not os.path.exists(os.path.join(cachedir, f"pixel_mask.pt")):

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

            if not no_cache:
                torch.save(batched_renderings.cpu(), os.path.join(cachedir, f"renders.pt"))
                torch.save(depth.cpu(), os.path.join(cachedir, f"depth.pt"))
                torch.save(pixel_mask.cpu(), os.path.join(cachedir, f"pixel_mask.pt"))
                torch.save(pixel_coords.cpu(), os.path.join(cachedir, f"pixel_coords.pt"))
                if use_normal_map and normal_batched_renderings is not None:
                    torch.save(normal_batched_renderings.cpu(), os.path.join(cachedir, f"normal_renders.pt"))

            # Move to CPU to save memory
            depth = depth.cpu()
            mesh_vertices = mesh_vertices.detach().cpu()
            mesh_faces = mesh_faces.detach().cpu()
            pixel_mask = pixel_mask.detach().cpu()
            pixel_coords = pixel_coords.detach().cpu()

            gc.collect()
            torch.cuda.empty_cache()
        else:
            batched_renderings = torch.load(os.path.join(cachedir, f"renders.pt"), map_location=device, weights_only=True)
            depth = torch.load(os.path.join(cachedir, f"depth.pt"), map_location=device, weights_only=True)
            pixel_coords = torch.load(os.path.join(cachedir, f"pixel_coords.pt"), map_location='cpu', weights_only=True)
            pixel_mask = torch.load(os.path.join(cachedir, f"pixel_mask.pt"), map_location='cpu', weights_only=True)

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
        if use_sam > 0:
            print("Generating segmentation masks using SAM ...")
            if timing:
                presamtime = time.time()

            from sam_utils import generate_sam_masks
            from GLOBALS import patch_sizes

            batched_sam = generate_sam_masks(
                batched_renderings, H, W, patch_sizes[model], device,
                debug=debug, debug_dir=cachedir,
            )

            if timing:
                samtime = time.time() - presamtime
                print(f"SAM segmentation completed in {samtime:02f} seconds")

        # ==================== Feature Extraction ====================
        if not os.path.exists(os.path.join(cachedir, f"view_features.pt")):
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
                feat_kwargs['prompt'] = description

            if model == "sam2":
                feat_kwargs['concat_hr'] = sam2_hr

            # NOTE: We do not concatenate extract features because memory cost too high
            view_features = extractor.get_pixel_features(
                batched_renderings, H, W, **feat_kwargs
            )

            # Apply SAM feature refinement if applicable
            if use_sam > 0 and extractor.supports_sam_refinement:
                from sam_utils import apply_sam_feature_refinement
                apply_sam_feature_refinement(
                    view_features, batched_sam, pixel_mask,
                    threshold=use_sam,
                    featurebatchsize=view_features[0].shape[0],
                    H=H, W=W,
                    debug=debug,
                    debug_dir=cachedir,
                )

            extractor.cleanup()

            # Clean up rendering data
            del batched_renderings
            if 'depth' in dir():
                del depth

            if timing:
                featuretime = time.time() - prefeaturetime
                print(f"Feature extraction complete in {featuretime/60} minutes")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            preloadtime = time.time()
            view_features = torch.load(os.path.join(cachedir, f"view_features.pt"), map_location='cpu', weights_only=True)
            print(f"Loaded cached view features from {os.path.join(cachedir, 'view_features.pt')}")

            if debug:
                t2 = (time.time() - preloadtime) / 60
                print(f"Time to load view features: {t2} minutes")

        if timing:
            pretraintime = time.time()

        # ==================== MLP Training ====================
        mesh_vertices = mesh_vertices.cpu()
        mesh_faces = mesh_faces.cpu()

        dataset = FeatureDataset(view_features, pixel_coords.detach().cpu(), pixel_mask.detach().cpu(),
                                 flatten=flattenviews, featurebatchsize=featurebatchsize)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True,
                                pin_memory=True, num_workers=2)

        from featuremlp import FeatureMLP

        ndim = view_features[0].shape[-1]
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
        scaler = torch.amp.GradScaler('cuda')

        from tqdm import tqdm

        for iteri in tqdm(range(startiter, iters)):
            totloss = 0

            # Subset dataloader if set
            if subsetepoch > 0:
                from torch.utils.data import DataLoader, Subset
                dataset_size = len(dataset)
                indices = list(range(dataset_size))
                np.random.shuffle(indices)
                subset_size = int(np.floor(dataset_size * subsetepoch))
                subset_indices = indices[:subset_size]
                random_subset = Subset(dataset, subset_indices)
                subset_loader = DataLoader(random_subset, batch_size=batchsize, shuffle=False)

            for i, (gtfeatures, positions, masks) in enumerate(tqdm(subset_loader if subsetepoch > 0 else dataloader)):
                gtfeatures = gtfeatures.view(-1, gtfeatures.shape[-1]).to(device, non_blocking=True)
                positions = positions.view(-1, 3).to(device, non_blocking=True)

                if not flattenviews and masks.numel() > 1:
                    masks = masks.view(-1).to(device, non_blocking=True)
                    gtfeatures = gtfeatures[masks]
                    positions = positions[masks]

                # Sample noise
                if noisen > 0:
                    n = len(positions)
                    # Sample 5D Gaussian noise, use 3 dims for spatial noise, all 5 for normalization
                    raw = torch.randn(n, noisen, 5, device=device)
                    norm = raw.norm(dim=-1, keepdim=True) / noiseradius  # (n, noisen, 1)
                    noise = raw[..., 2:] / norm  # use last 3 dims as spatial noise (n, noisen, 3)
                    noisy_positions = positions.unsqueeze(1) + noise
                    positions = torch.cat([positions.unsqueeze(1), noisy_positions], dim=1).reshape(-1, 3)
                    gtfeatures = gtfeatures.repeat_interleave(noisen + 1, dim=0)

                with torch.amp.autocast('cuda'):
                    pred_features = encoder(positions)
                    batch_loss = torch.nn.MSELoss(reduction='sum')(pred_features, gtfeatures.float())

                optim.zero_grad(set_to_none=True)
                scaler.scale(batch_loss).backward()
                scaler.step(optim)
                scaler.update()

                totloss += batch_loss.item()

            totloss /= len(dataloader)
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

        if not no_cache:
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
    parse.add_argument('-d', "--description", nargs="+", type=str, default=[], help="text description for diff3f encoding")
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
    parse.add_argument("--model", choices={"dino2", "dino3", "diff3f", 'clip', 'sam', 'sam2', 'radio'}, type=str, default="dino2")
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
    parse.add_argument('--flattenviews', action="store_true", help='flatten the view features')
    parse.add_argument("--batchsize", type=int, default=2, help="views to batch OR pixels to batch during feature optimization")
    parse.add_argument("--viewbatchsize", type=int, default=16, help="number of views to batch for rendering")
    parse.add_argument("--featurebatchsize", type=int, default=2, help="number of views to batch for feature extraction")
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--iters", type=int, default=25)

    ### Gaussian blurring
    parse.add_argument("--noiseradius", type=float, default=0.05, help="maximum radius for sampling outside of the vertex")
    parse.add_argument("--noisen", type=int, default=0, help="noise samples for every vertex")

    ### Subset epoch training
    parse.add_argument('--subsetepoch', type=float, default=0, help="If > 0, will only train on a random subset of this fraction of the dataset every epoch.")

    ### SAM feature reassignment
    parse.add_argument('--use_sam', type=float, default=0, help="SAM segmentation to fix feature bleeding. Value determines pixel features which are outliers from the cluster mode.")

    ### MLP parameters
    parse.add_argument('--nlayers', type=int, default=4)
    parse.add_argument('--width', type=int, default=256)
    parse.add_argument('--positional_encoding', action="store_true", help="using fourier features for positional encoding")
    parse.add_argument('--sigma', type=float, default=5.0, help="sigma for fourier features")

    args = parse.parse_args()
    barycentric_distillation(**vars(args))
