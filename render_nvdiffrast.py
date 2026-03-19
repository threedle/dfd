"""
nvdiffrast-based batch rendering pipeline.
Replaces both render.py (PyTorch3D) and pyrendering.py (PyRender) from the original codebase.
"""
import torch
import numpy as np
import nvdiffrast.torch as dr

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


def _perspective_projection(fov_y, aspect, near, far):
    """
    Compute a perspective projection matrix.

    Args:
        fov_y: vertical field of view in radians
        aspect: width / height
        near: near clipping plane
        far: far clipping plane

    Returns:
        proj: (4, 4) numpy array, projection matrix
    """
    t = np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = 1.0 / (aspect * t)
    proj[1, 1] = 1.0 / t
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def generate_cameras(num_views, viewtype, radius):
    """
    Generate camera eye positions and view matrices for the given view configuration.

    Args:
        num_views: number of views to generate
        viewtype: 'fib' (fibonacci), 'cube' (6 views), or grid-based
        radius: camera distance from origin

    Returns:
        eye_positions: (N, 3) numpy array of camera positions
        view_matrices: (N, 4, 4) numpy array of world-to-camera transforms
        num_views: actual number of views (may differ for 'cube')
    """
    # Default
    if viewtype == "default":
        elev_steps = 3
        azim_steps = num_views // elev_steps
        num_views = elev_steps * azim_steps
        elev_start = -15
        elev_end = 15
        azim_start = 0
        azim_end = 360 - 360 / azim_steps

        elevation_deg = np.linspace(elev_start, elev_end, elev_steps)
        azimuth_deg = np.linspace(azim_start, azim_end, azim_steps)

        # Create grid: for each azimuth, repeat all elevations
        azimuth_grid = np.repeat(azimuth_deg, elev_steps)
        elevation_grid = np.tile(elevation_deg, azim_steps)

        elev_rad = np.radians(elevation_grid)
        azim_rad = np.radians(azimuth_grid)
        eye_positions = radius * np.stack([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
        ], axis=1)
    elif viewtype == 'fib':
        goldenRatio = (1 + 5**0.5) / 2
        i = np.arange(0, num_views)
        azimuth = 2 * np.pi * i / goldenRatio
        elevation = np.arccos(1 - 2 * (i + 0.5) / num_views)
        eye_positions = radius * np.stack([
            np.cos(azimuth) * np.sin(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation)
        ], axis=1)
    elif viewtype == 'cube':
        num_views = 6
        # 6 axis-aligned directions
        elevation_deg = np.array([0., 0., 90., -90., 0., 0.])
        azimuth_deg = np.array([0., 180., 0., 0., 90., -90.])
        elev_rad = np.radians(elevation_deg)
        azim_rad = np.radians(azimuth_deg)
        eye_positions = radius * np.stack([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
        ], axis=1)

    target = np.array([0., 0., 0.])
    up = np.array([0., 1., 0.])

    view_matrices = np.stack([
        _look_at(eye_positions[i], target, up)
        for i in range(num_views)
    ], axis=0)

    return eye_positions, view_matrices, num_views


def _compute_vertex_normals(vertices, faces):
    """
    Compute per-vertex normals by averaging face normals weighted by face area.

    Args:
        vertices: (V, 3) tensor
        faces: (F, 3) int tensor

    Returns:
        vertex_normals: (V, 3) tensor, unit normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)

    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)

    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
    return vertex_normals


def phong_shade(normals, light_dir, vertex_colors, ambient=(0.6, 0.6, 0.6), diffuse=(0.4, 0.4, 0.5)):
    """
    Apply Lambertian + ambient shading (no specular, matching original HardPhongShader settings).

    Args:
        normals: (B, H, W, 3) interpolated surface normals
        light_dir: (B, 1, 1, 3) normalized direction from surface to light
        vertex_colors: (B, H, W, 3) base vertex colors
        ambient: tuple of 3 floats
        diffuse: tuple of 3 floats

    Returns:
        shaded: (B, H, W, 3) shaded colors in [0, 1]
    """
    ambient_t = torch.tensor(ambient, device=normals.device, dtype=normals.dtype).reshape(1, 1, 1, 3)
    diffuse_t = torch.tensor(diffuse, device=normals.device, dtype=normals.dtype).reshape(1, 1, 1, 3)

    # Normalize normals
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # Lambertian: max(dot(N, L), 0)
    cos_angle = torch.clamp(torch.sum(normals * light_dir, dim=-1, keepdim=True), min=0.0) # B, H, W, 1

    # Final color
    shaded = vertex_colors * (ambient_t + diffuse_t * cos_angle) # B, H, W, 3
    return torch.clamp(shaded, 0.0, 1.0)


@torch.no_grad()
def run_rendering(glctx, device, vertices, faces, num_views, H, W,
                  vertex_colors=None, vertex_normals=None,
                  use_normal_map=False, radius=1, viewtype='default',
                  texture_data=None, batch_size=16):
    """
    Core rendering function using nvdiffrast.

    Args:
        glctx: nvdiffrast rasterization context
        device: torch device
        vertices: (V, 3) float tensor, mesh vertices (already normalized)
        faces: (F, 3) int32 tensor, triangle face indices
        num_views: number of views
        H, W: image resolution
        vertex_colors: (V, 3) float tensor or None
        vertex_normals: (V, 3) float tensor or None
        use_normal_map: whether to produce normal map renderings
        radius: camera distance
        viewtype: 'default', 'fib', 'cube'
        texture_data: optional dict with 'uv_coords', 'uv_faces', 'texture_image' for UV-textured meshes

    Returns:
        batched_renderings: (N, H, W, 4) float tensor (RGBA)
        normal_batched_renderings: (N, H, W, 4) float tensor or None
        depth: (N, H, W, 1) float tensor
        pixel_mask: (N, H, W) bool tensor
        pixel_coords: (N, H, W, 3) float tensor, world-space coordinates per pixel
    """
    eye_positions, view_matrices, num_views = generate_cameras(
        num_views, viewtype, radius
    )

    # Update num_views to be the actual number of views
    num_views = len(eye_positions)

    # Compute projection matrix
    fov_y = np.radians(60)
    aspect = W / H
    near = 0.01
    far = 100.0
    proj_np = _perspective_projection(fov_y, aspect, near, far)
    proj_np = np.tile(proj_np, (min(batch_size, num_views), 1, 1))  # (B, 4, 4) numpy, each view has its own projection matrix

    # Prepare face indices for nvdiffrast (int32)
    faces_int = faces.to(torch.int32).to(device)

    if vertex_normals is None:
        vertex_normals = _compute_vertex_normals(vertices.to(device), faces.long().to(device))
    vertex_normals = vertex_normals.to(device)

    # Default vertex colors
    if vertex_colors is None:
        vertex_colors = torch.ones_like(vertices) * 0.8
    vertex_colors = vertex_colors.to(device).float()

    # Ensure vertices are on device
    vertices_dev = vertices.to(device).float()

    # Prepare attributes for interpolation: position (3) + color (3) + normal (3) = 9
    vertex_attrs = torch.cat([vertices_dev, vertex_colors, vertex_normals], dim=1)  # (V, 9)

    # Pre-compute homogeneous vertices and projection tensor (shared across batches)
    ones = torch.ones(vertices_dev.shape[0], 1, device=device)
    verts_hom = torch.cat([vertices_dev, ones], dim=1)  # (V, 4)
    proj_t = torch.from_numpy(proj_np).float().to(device)  # (max_B, 4, 4)
    eye_positions_t = torch.from_numpy(eye_positions).float().to(device)  # (N, 3)

    # Pre-move texture data to device once
    if texture_data is not None:
        uv_coords_dev = texture_data['uv_coords'].to(device).float()       # (V_uv, 2)
        uv_faces_dev = texture_data['uv_faces'].to(torch.int32).to(device)  # (F, 3)
        texture_image_dev = texture_data['texture_image'].to(device).float() # (1, tex_H, tex_W, 3)

    # Process each view
    all_renderings = []
    all_normal_renderings = []
    all_depth = []
    all_pixel_mask = []
    all_pixel_coords = []

    for batch_start in range(0, num_views, batch_size):
        # Build MVP matrix for this view
        view_mat = view_matrices[batch_start:batch_start+batch_size]  # (B, 4, 4) numpy
        B = len(view_mat)
        view_t = torch.from_numpy(view_mat).float().to(device)
        mvp_t = proj_t[:B] @ view_t  # (B, 4, 4)

        # Transform vertices to clip space
        clip_verts = (mvp_t @ verts_hom.T.unsqueeze(0)).permute(0, 2, 1).contiguous()  # (B, V, 4)

        # Rasterize
        # NOTE: Confirmed that perspective division happens internally.
        rast_out, _ = dr.rasterize(glctx, clip_verts, faces_int, resolution=[H, W])
        # rast_out: (B, H, W, 4) -> [u, v, w-normalized depth, face_id (1-indexed)]

        # Extract mask
        face_id = rast_out[..., 3:4]  # (B, H, W, 1)
        mask = (face_id > 0).squeeze(-1)  # (B, H, W)

        # Interpolate vertex attributes
        attrs_unsqueezed = vertex_attrs.unsqueeze(0)  # (B, V, 9)
        interp_out, _ = dr.interpolate(attrs_unsqueezed, rast_out, faces_int)
        # interp_out: (B, H, W, 9)

        # Split interpolated attributes
        pixel_world_pos = interp_out[..., :3]   # (B, H, W, 3)
        pixel_colors = interp_out[..., 3:6]     # (B, H, W, 3)
        pixel_normals = interp_out[..., 6:9]    # (B, H, W, 3)

        # Interpolate depth values
        depth_map, _ = dr.interpolate(clip_verts[..., 2:3].contiguous(), rast_out, faces_int)

        # Apply Phong shading
        eye_pos = eye_positions_t[batch_start:batch_start+B]
        light_dir = torch.nn.functional.normalize(
            eye_pos.reshape(B, 1, 1, 3) - pixel_world_pos, dim=-1
        )

        # Handle UV textures if provided
        if texture_data is not None:
            texture_image = texture_image_dev.expand(B, -1, -1, -1)  # (B, tex_H, tex_W, 3) - no copy
            uv_attrs = uv_coords_dev.unsqueeze(0).expand(B, -1, -1)  # (B, V_uv, 2) - no copy
            uv_interp, _ = dr.interpolate(uv_attrs.contiguous(), rast_out, uv_faces_dev)  # (B, H, W, 2)

            # nvdiffrast texture expects (B, H, W, C) texture and (B, H, W, 2) uv
            pixel_colors = dr.texture(texture_image.contiguous(), uv_interp)  # (B, H, W, 3)

        shaded = phong_shade(pixel_normals, light_dir, pixel_colors)

        # Add alpha channel
        alpha = mask.unsqueeze(-1).float()
        rgba = torch.cat([shaded, alpha], dim=-1)  # (B, H, W, 4)

        # Replace background with white
        rgba[~mask.unsqueeze(-1).expand_as(rgba)] = 1
        # rgba[~mask.unsqueeze(-1).expand_as(rgba)] = 0
        pixel_world_pos[~mask.unsqueeze(-1).expand_as(pixel_world_pos)] = 0

        # Flip from OpenGL (bottom-up) to image (top-down) convention
        rgba = torch.flip(rgba, dims=(1,))
        depth_map = torch.flip(depth_map, dims=(1,))
        flip_mask = torch.flip(mask, dims=(1,))
        pixel_world_pos = torch.flip(pixel_world_pos, dims=(1,))

        all_renderings.append(rgba)
        all_depth.append(depth_map)
        all_pixel_mask.append(flip_mask)
        all_pixel_coords.append(pixel_world_pos)

        # Normal map rendering
        if use_normal_map:
            # Render normals as colors (map from [-1,1] to [0,1])
            normal_colors = (pixel_normals * 0.5 + 0.5) * mask.unsqueeze(-1).float()
            normal_rgba = torch.cat([normal_colors, alpha], dim=-1)
            normal_rgba = torch.flip(normal_rgba, dims=(1,))
            all_normal_renderings.append(normal_rgba)

    # Stack all views
    batched_renderings = torch.cat(all_renderings, dim=0)
    depth = torch.cat(all_depth, dim=0)                            # (N, H, W, 1)
    pixel_mask = torch.cat(all_pixel_mask, dim=0)                  # (N, H, W)
    pixel_coords = torch.cat(all_pixel_coords, dim=0)              # (N, H, W, 3)

    normal_batched_renderings = None
    if use_normal_map and len(all_normal_renderings) > 0:
        normal_batched_renderings = torch.cat(all_normal_renderings, dim=0)

    return batched_renderings, normal_batched_renderings, depth, pixel_mask, pixel_coords