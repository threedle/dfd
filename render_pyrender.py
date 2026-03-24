import torch
import pyrender
import numpy as np

def _look_at(eye, target, up):
    """
    Compute a look-at view matrix (world-to-camera).

    Args:
        eye: (3,) numpy array, camera position
        target: (3,) numpy array, look-at point
        up: (3,) numpy array, up direction

    Returns:
        view_matrix: (4, 4) numpy array, world-to-camera transform
    """
    forward = eye - target
    forward = forward / np.linalg.norm(forward)

    # Handle degenerate case where forward is parallel to up
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0])

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up = new_up / np.linalg.norm(new_up)

    # View matrix: rotation part
    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = new_up
    R[2, :3] = forward

    # Translation: dot products for camera-space translation
    T = np.eye(4)
    T[:3, 3] = -eye

    return R @ T

def _generate_eye_positions(num_views, viewtype, radius):
    if viewtype == "default":
        elev_steps = 3
        azim_steps = max(1, num_views // elev_steps)
        num_views_out = elev_steps * azim_steps
        elev_start, elev_end = -15.0, 15.0
        azim_start = 0.0
        azim_end = 360.0 - 360.0 / azim_steps

        elevation_deg = np.linspace(elev_start, elev_end, elev_steps)
        azimuth_deg = np.linspace(azim_start, azim_end, azim_steps)

        azimuth_grid = np.repeat(azimuth_deg, elev_steps)
        elevation_grid = np.tile(elevation_deg, azim_steps)

        elev_rad = np.radians(elevation_grid)
        azim_rad = np.radians(azimuth_grid)
        eye_positions = radius * np.stack([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
        ], axis=1)
        return eye_positions, num_views_out

    elif viewtype == "fib":
        goldenRatio = (1 + 5**0.5) / 2
        i = np.arange(0, num_views)
        azimuth = 2 * np.pi * i / goldenRatio
        elevation = np.arccos(1 - 2 * (i + 0.5) / num_views)
        eye_positions = radius * np.stack([
            np.cos(azimuth) * np.sin(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation)
        ], axis=1)
        return eye_positions, num_views
    # else:
    #     # cube
    #     num_views_out = 6
    #     elevation_deg = np.array([0., 0., 90., -90., 0., 0.])
    #     azimuth_deg = np.array([0., 180., 0., 0., 90., -90.])
    #     elev_rad = np.radians(elevation_deg)
    #     azim_rad = np.radians(azimuth_deg)
    #     eye_positions = radius * np.stack([
    #         np.cos(elev_rad) * np.cos(azim_rad),
    #         np.sin(elev_rad),
    #         np.cos(elev_rad) * np.sin(azim_rad),
    #     ], axis=1)
    #     return eye_positions, num_views_out

@torch.no_grad()
def run_rendering(device, mesh, num_views, H, W,
                  radius=1, viewtype='default'):
    """
    Pyrender GLB loaded with trimesh.

    Note:
    - Normal-map rendering is best-effort: it is returned if PyRender supports a normals pass; otherwise None.
    """
    # Local imports to keep module import light-weight in headless environments.
    import os
    import trimesh

    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # -------------------- Camera generation (match render_nvdiffrast) --------------------
    eye_positions, num_views = _generate_eye_positions(num_views, viewtype, radius)

    # -------------------- Build PyRender mesh --------------------
    mesh_pr = pyrender.Mesh.from_trimesh(mesh)

    # -------------------- Intrinsics (match 60deg fov in render_nvdiffrast) --------------------
    fov_y = np.radians(60.0)
    aspect = float(W) / float(H)
    fov_x = 2.0 * np.arctan(np.tan(fov_y / 2.0) * aspect)
    fx = (W / 2.0) / np.tan(fov_x / 2.0)
    fy = (H / 2.0) / np.tan(fov_y / 2.0)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    pyrender_camera = pyrender.IntrinsicsCamera(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)
    )

    # -------------------- Render loop --------------------
    r = pyrender.OffscreenRenderer(W, H)

    all_rgba = []
    all_depth = []
    all_mask = []
    all_pixel_coords = []

    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    # Matches the original coordinate conversion used in this file.
    T_OPENGL_TO_OPENCV = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )

    for viewi in range(len(eye_positions)):
        scene = pyrender.Scene(ambient_light=(0.6, 0.6, 0.6))
        scene.add(mesh_pr)

        w2c = _look_at(eye_positions[viewi], target, up)
        camera_pose = np.linalg.inv(w2c) # pose is c2w
        scene.add(pyrender_camera, pose=camera_pose)

        light = pyrender.PointLight(color=(0.4, 0.4, 0.5), intensity=1.0)
        scene.add(light, pose=camera_pose)

        flags = pyrender.RenderFlags.RGBA
        color_rgba_u8, depth_hw = r.render(scene, flags=flags)

        rgba = (color_rgba_u8.astype(np.float32) / 255.0)
        depth = depth_hw.astype(np.float32)  # (H, W)
        mask = depth > 0

        # Match render_nvdiffrast: white background
        rgba[~mask] = 1.0

        # Pixel -> world coordinates
        good_y, good_x = np.where(mask)
        z = depth[good_y, good_x]
        normalized_x = good_x.astype(np.float32) - float(cx)
        normalized_y = good_y.astype(np.float32) - float(cy)

        pc_x = normalized_x * z / float(fx)
        pc_y = normalized_y * z / float(fy)
        pc_z = z

        pc = np.vstack((pc_x, pc_y, pc_z)).T
        pc_h = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32)))
        cam_pose_z_opencv = (camera_pose @ T_OPENGL_TO_OPENCV).astype(np.float32)
        world_coords = (cam_pose_z_opencv @ pc_h.T).T[:, :3]

        pixel_coords = np.zeros((H, W, 3), dtype=np.float32)
        pixel_coords[mask] = world_coords

        all_rgba.append(rgba)
        all_depth.append(depth[..., None])  # (H, W, 1)
        all_mask.append(mask)
        all_pixel_coords.append(pixel_coords)

    r.delete()

    batched_renderings = torch.from_numpy(np.stack(all_rgba, axis=0)).float().to(device)          # (N, H, W, 4)
    depth_t = torch.from_numpy(np.stack(all_depth, axis=0)).float().to(device)                   # (N, H, W, 1)
    pixel_mask = torch.from_numpy(np.stack(all_mask, axis=0)).bool().to(device)                  # (N, H, W)
    pixel_coords = torch.from_numpy(np.stack(all_pixel_coords, axis=0)).float().to(device)       # (N, H, W, 3)

    return batched_renderings, depth_t, pixel_mask, pixel_coords