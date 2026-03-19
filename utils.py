import torch
import numbers
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
import colorsys

def normalize(v):
    from igl import bounding_box
    bb_vs, bf = bounding_box(v)
    v -= np.mean(bb_vs, axis=0)
    v /= np.max(np.linalg.norm(v, axis=1))

    return v

def cube_normalize(v):
    from igl import bounding_box
    bb_vs, bf = bounding_box(v)
    v -= np.mean(bb_vs, axis=0)
    bbox_length = np.max(np.max(bb_vs, axis=0) - np.min(bb_vs, axis=0)) / 2
    v /= bbox_length

    return v

def get_face_angles(vertices, faces, eps=1e-12):
    # Fast face area calculation
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    a = v1 - v0
    b = v2 - v0
    c = v2 - v1
    d = v0 - v1
    e = v0 - v2
    f = v1 - v2

    def angle(u, v):
        dot = np.einsum('ij,ij->i', u, v)
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        cos = dot / (np.maximum(nu * nv, eps))
        return np.arccos(np.clip(cos, -1.0, 1.0))

    return np.stack([
        angle(a, b),   # at v0
        angle(c, d),   # at v1
        angle(e, f)    # at v2
    ], axis=1)

def get_edges(faces):
    # Get edges in a consistent way
    edges = []
    for f in faces:
        edges.extend([(f[i], f[(i+1)%3]) for i in range(3)])

    return np.array(edges)

def get_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0

    cx = e0[:, 1]*e1[:, 2] - e0[:, 2]*e1[:, 1]
    cy = e0[:, 2]*e1[:, 0] - e0[:, 0]*e1[:, 2]
    cz = e0[:, 0]*e1[:, 1] - e0[:, 1]*e1[:, 0]

    return 0.5 * np.sqrt(cx*cx + cy*cy + cz*cz)

def clear_directory(path):
    import os
    import shutil
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


#### Image processing ####
# Find the whitespace/alpha bounding box
def find_bounding_boxes(images):
    """ images: nested list of PIL images; we assume every list is the SAME LENGTH """
    cols = len(images[0])
    rows = len(images)

    width_bboxes = [None] * cols
    height_bboxes = [None] * rows
    height_bbox = height_bboxes[rowi]
    for rowi, row in enumerate(images):
        height_bbox = height_bboxes[rowi]

        for coli, img in enumerate(row):
            width_bbox = width_bboxes[col]

            if img.mode == 'RGBA':
                bbox = img.getbbox()

                if height_bbox is None:
                    height_bbox = (bbox[1], bbox[3])
                else:
                    height_bbox = (
                        min(height_bbox[0], bbox[1]),
                        max(height_bbox[1], bbox[3])
                    )
                height_bboxes[rowi] = height_bbox

                if width_bbox is None:
                    width_bbox = (bbox[0], bbox[2])
                else:
                    width_bbox = (
                        min(width_bbox[0], bbox[0]),
                        max(width_bbox[1], bbox[2])
                    )
                width_bboxes[coli] = width_bbox

            else:
                # Convert white pixels to zero alpha
                img = img.convert('RGBA')
                np_img = np.asarray(img)
                np_img[:, :, 3] = (255 * (np_img[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
                img = Image.fromarray(np_img, 'RGBA')

                # Replace the image with the modified one
                row[coli] = img

                bbox = img.getbbox()

                if height_bbox is None:
                    height_bbox = (bbox[1], bbox[3])
                else:
                    height_bbox = (
                        min(height_bbox[0], bbox[1]),
                        max(height_bbox[1], bbox[3])
                    )
                height_bboxes[rowi] = height_bbox

                if width_bbox is None:
                    width_bbox = (bbox[0], bbox[2])
                else:
                    width_bbox = (
                        min(width_bbox[0], bbox[0]),
                        max(width_bbox[1], bbox[2])
                    )
                width_bboxes[coli] = width_bbox

#### Bezier stuff #####
def sample_bezier(curves, t, return_tangent=False, return_t=False):
    """ Sample bezier function
    curves: B x 4 x 3 or 4 x 3
    t: B or scalar """

    if len(curves.shape) == 2:
        assert type(t) == int
        sample_t = torch.linspace(0, 1, t).to(curves.device)

        w0 = (1 - sample_t)**3
        w1 = 3 * (1 - sample_t)**2 * sample_t
        w2 = 3 * (1 - sample_t) * sample_t**2
        w3 = sample_t**3

        samples = w0.reshape(-1, 1) * curves[None, 0] + \
                    w1.reshape(-1, 1) * curves[None, 1] + \
                    w2.reshape(-1, 1) * curves[None, 2] + \
                        w3.reshape(-1, 1) * curves[None, 3] # t x 3

        if return_tangent:
            w0_tangent = 3 * (1 - sample_t)**2
            w1_tangent = 6 * (1 - sample_t) * sample_t
            w2_tangent = 3 * sample_t**2

            tangent_samples = w0_tangent.reshape(-1, 1) * (curves[None, 1] - curves[None, 0]) + \
                        w1_tangent.reshape(-1, 1) * (curves[None, 2] - curves[None, 1]) + \
                        w2_tangent.reshape(-1, 1) * (curves[None, 3] - curves[None, 2])
            tangent_samples /= torch.linalg.norm(tangent_samples, dim=1, keepdim=True)
    else:
        if type(t) == int:
            sample_t = torch.linspace(0, 1, t).to(curves.device)
            w0 = (1 - sample_t)**3
            w1 = 3 * (1 - sample_t)**2 * sample_t
            w2 = 3 * (1 - sample_t) * sample_t**2
            w3 = sample_t**3

            samples = w0.reshape(1, -1, 1) * curves[:, None, 0] + \
                        w1.reshape(1, -1, 1) * curves[:, None, 1] + \
                        w2.reshape(1, -1, 1) * curves[:, None, 2] + \
                            w3.reshape(1, -1, 1) * curves[:, None, 3] # B x t x 3

            if return_tangent:
                w0_tangent = 3 * (1 - sample_t)**2
                w1_tangent = 6 * (1 - sample_t) * sample_t
                w2_tangent = 3 * sample_t**2

                tangent_samples = w0_tangent.reshape(1, -1, 1) * (curves[:, None, 1] - curves[:, None, 0]) + \
                            w1_tangent.reshape(1, -1, 1) * (curves[:, None, 2] - curves[:, None, 1]) + \
                            w2_tangent.reshape(1, -1, 1) * (curves[:, None, 3] - curves[:, None, 2]) # B x t x 3

                tangent_samples /= torch.linalg.norm(tangent_samples, dim=2, keepdim=True)
        else:
            assert len(t) == len(curves)

            samples = []
            tangent_samples = []
            sample_t = []
            for curvei in range(len(t)):
                tmp_t = torch.linspace(0, 1, t[curvei]).to(curves.device)
                sample_t.append(tmp_t)

                w0_sample = (1 - tmp_t)**3
                w1_sample = 3 * (1 - tmp_t)**2 * tmp_t
                w2_sample = 3 * (1 - tmp_t) * tmp_t**2
                w3_sample = tmp_t**3

                tmpsamples = w0_sample.reshape(-1, 1) * curves[curvei, None, 0] + \
                    w1_sample.reshape(-1, 1) * curves[curvei, None, 1] + \
                    w2_sample.reshape(-1, 1) * curves[curvei, None, 2] + \
                        w3_sample.reshape(-1, 1) * curves[curvei, None, 3] # t x 3
                samples.append(tmpsamples)

                if return_tangent:
                    w0_tangent = 3 * (1 - tmp_t)**2
                    w1_tangent = 6 * (1 - tmp_t) * tmp_t
                    w2_tangent = 3 * tmp_t**2

                    tmp_tangent = w0_tangent.reshape(-1, 1) * (curves[curvei, None, 1] - curves[curvei, None, 0]) + \
                                w1_tangent.reshape(-1, 1) * (curves[curvei, None, 2] - curves[curvei, None, 1]) + \
                                w2_tangent.reshape(-1, 1) * (curves[curvei, None, 3] - curves[curvei, None, 2])
                    tmp_tangent /= torch.linalg.norm(tmp_tangent, dim=1, keepdim=True)

                    tangent_samples.append(tmp_tangent)

            # samples = torch.nested.nested_tensor(samples)
            # tangent_samples = torch.nested.nested_tensor(tangent_samples)
            # sample_t = torch.nested.nested_tensor(sample_t)

    if return_tangent and return_t:
        return samples, tangent_samples, sample_t
    elif return_tangent:
        return samples, tangent_samples
    elif return_t:
        return samples, sample_t
    else:
        return samples

# Approximates cubic bezier length using piecewise linear approximation
def bcurve_length(points, num_samples=1000):
    """
        points: B x 4 x 3 (torch.tensor) batched or unbatched
        num_samples: int
    """
    # Bezier weighting
    sample_t = torch.linspace(0, 1, num_samples).to(points.device)

    # NOTE: bezier interpolation follows binomial distribution so the weights will always sum to 1
    w0 = (1 - sample_t)**3
    w1 = 3 * (1 - sample_t)**2 * sample_t
    w2 = 3 * (1 - sample_t) * sample_t**2
    w3 = sample_t**3

    if len(points.shape) == 2:
        # Unbatched
        samples = w0[:, None] * points[None, 0] + w1[:, None] * points[None, 1] + \
                    w2[:, None] * points[None, 2] + w3[:, None] * points[None, 3] # N x 3
        length = torch.linalg.norm(samples[1:] - samples[:-1], dim=1).sum()
    else:
        samples = w0.reshape(1, -1, 1) * points[:, None, 0] + \
                    w1.reshape(1, -1, 1) * points[:, None, 1] + \
                    w2.reshape(1, -1, 1) * points[:, None, 2] + \
                        w3.reshape(1, -1, 1) * points[:, None, 3]
        length = torch.linalg.norm(samples[:, 1:] - samples[:, :-1], dim=2).sum(dim=1)

    return length

def gen_elev_azim(elev_1, elev_2, elev_n, azim_1, azim_2, azim_n,
                  center_elev=None, center_azim=None,
                  device=torch.device('cpu')):

    if center_elev is not None and center_azim is not None:
        elev = []
        azim = []

        for ce, ca in zip(center_elev, center_azim):
            elev.append(torch.linspace(ce + elev_1, ce + elev_2, elev_n).repeat_interleave(azim_n).float().to(device))
            azim.append(torch.linspace(ca + azim_1, ca + azim_2, azim_n).tile(elev_n).float().to(device))
        elev = torch.cat(elev)
        azim = torch.cat(azim)
    else:
        elev = torch.linspace(elev_1, elev_2, elev_n).repeat_interleave(azim_n).float().to(device)
        azim = torch.linspace(azim_1, azim_2, azim_n).tile((elev_n,)).float().to(device)
    return elev, azim

##### ROTATION STUFF ######
# Rotations about the standard axes
def get_rotation_x(angle):
    # Returns rotation matrix for rotation about x-axis by angle (in radians)
    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(angle), -torch.sin(angle)],
                         [0, torch.sin(angle), torch.cos(angle)]])

def get_rotation_y(angle):
    # Returns rotation matrix for rotation about y-axis by angle (in radians)
    return torch.tensor([[torch.cos(angle), 0, torch.sin(angle)],
                         [0, 1, 0],
                         [-torch.sin(angle), 0, torch.cos(angle)]])

def get_rotation_z(angle):
    # Returns rotation matrix for rotation about z-axis by angle (in radians)
    return torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                         [torch.sin(angle), torch.cos(angle), 0],
                         [0, 0, 1]])

def build_affine_matrix(rotation=None, translation=None, scale=None):
    # Builds an affine transformation matrix from rotation, translation, and scale
    # rotation: 3x3 rotation matrix or None
    # translation: 3-element vector or None
    # scale: 3-element vector or None

    if rotation is None:
        rotation = torch.eye(3)
    if translation is None:
        translation = torch.zeros(3)
    if scale is None:
        scale = torch.ones(3)

    affine_matrix = torch.eye(4)
    affine_matrix[:3, :3] = rotation * scale.unsqueeze(1)
    affine_matrix[:3, 3] = translation

    return affine_matrix

def get_orthogonal_vector(v):
    # Get an orthogonal vector to v
    # Algorith: https://math.stackexchange.com/questions/133177/finding-a-unit-vector-perpendicular-to-another-vector
    if torch.allclose(v, torch.zeros_like(v)):
        raise ValueError("Cannot get orthogonal vector to zero vector")

    m = torch.where(~torch.isclose(v, torch.tensor([0, 0, 0])))[0][0]
    n = (m + 1) % 3

    y = torch.zeros(3)
    y[m] = -v[n]
    y[n] = v[m]

    return y / torch.linalg.norm(y)

def get_cross_product_matrix(v):
    # From: https://wikimedia.org/api/rest_v1/media/math/render/svg/e3ddca93f49b042e6a14d5263002603fc0738308
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

def get_rotation_from_axis_and_angle(axis, angle):
    # From: https://en.wikipedia.org/wiki/Rotation_matrix#:~:text=Rotation%20matrix%20from%20axis%20and%20angle
    cp = get_cross_product_matrix(axis)
    return torch.cos(angle) * torch.eye(3) + torch.sin(angle) * cp + (1 - torch.cos(angle)) * torch.outer(axis, axis)

def get_rotation(v1, v2):
    # Get rotation matrix to rotate v1 to v2
    # NOTE: v1 v2 must be unit vectors

    # Batched
    if len(v1.shape) > 1 and len(v2.shape) > 1:
        v = torch.linalg.cross(v1, v2, dim=1)
        s = torch.linalg.norm(v, dim=1)
        c = torch.einsum('ij,ij->i', v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        antiparallel_mask = torch.isclose(c, torch.tensor(-1., device=c.device))
        if torch.any(antiparallel_mask):
            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1[antiparallel_mask])
            R_antiparallel = get_rotation_from_axis_and_angle(ortho, np.pi)
            R = torch.eye(3).repeat(v1.size(0), 1, 1)
            R[antiparallel_mask] = R_antiparallel
        else:
            R = torch.eye(3, device=v2.device).repeat(v2.size(0), 1, 1)

        # NOTE: When parallel, the answer is identity and is correct
        vx = torch.zeros((v2.size(0), 3, 3), device=v2.device)
        vx[:, 0, 1] = -v[:, 2]
        vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2]
        vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]
        vx[:, 2, 1] = v[:, 0]

        R[antiparallel_mask == False] += vx[antiparallel_mask == False] + torch.bmm(vx[antiparallel_mask == False], vx[antiparallel_mask == False]) * (1 / (1 + c[antiparallel_mask == False])).unsqueeze(1).unsqueeze(2)

        torch.testing.assert_close(torch.matmul(R, v1.unsqueeze(-1)).squeeze(), v2, rtol=1e-3, atol=1e-3)
    else:
        v = torch.linalg.cross(v1, v2)
        s = torch.linalg.norm(v)
        c = torch.dot(v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        if torch.allclose(c, torch.tensor(-1., device=c.device)):
            print("get_rotation: Antiparallel vectors detected")

            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1)
            return get_rotation_from_axis_and_angle(ortho, np.pi)

        # NOTE: When parallel, the answer is identity and is correct
        vx = torch.tensor([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

        R = torch.eye(3) + vx + vx @ vx * 1 / (1 + c)

        torch.testing.assert_close(torch.matmul(R, v1), v2, rtol=1e-5, atol=1e-5)

    return R

def get_pos_from_elev(elev, azim, r=3.0, origin=torch.zeros(3), origin_vector=None,
                      device=torch.device('cpu'), blender=False):
    """
    Convert tensor elevation/azimuth values into camera projections (with respect to origin/origin_vector)

    Base conversion assumes (1,0,0) vector as the origin vector.

    Args:
        elev (torch.Tensor): elevation
        azim (torch.Tensor): azimuth
        r (float, optional): radius. Defaults to 3.0.

    Returns:
        camera position vectors
    """
    if blender:
        # Y and Z axes are swapped, and rotation is opposite direction
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.cos(elev) * torch.sin(-azim)
        z = r * torch.sin(elev)
    else:
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)

    if len(x.shape) == 0:
        pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
    else:
        pos = torch.stack([x, y, z], dim=1).to(device)

    # Apply rotation matrix to origin vector
    if origin_vector is not None:
        rotation_matrix = get_rotation(torch.tensor([1., 0., 0.], device=device), origin_vector.to(device))
        pos = torch.mm(rotation_matrix, pos.T).T

    return pos + origin.to(device)

def generate_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 1
    value = 1
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors

# def plot_mesh(myMesh,cmap=None):
#     mp.plot(myMesh.vert, myMesh.face,c=cmap)

# def double_plot(myMesh1,myMesh2,cmap1=None,cmap2=None):
#     d = mp.subplot(myMesh1.vert, myMesh1.face, c=cmap1, s=[2, 2, 0])
#     mp.subplot(myMesh2.vert, myMesh2.face, c=cmap2, s=[2, 2, 1], data=d)

def get_colors(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy()"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError


def cdist(a, b):
    if len(a) > 30000:
        return cdist_batch(a, b, batch_size=30000)

    dist = torch.cdist(a, b)
    return dist

def cdist_batch(a, b, batch_size=30000):
    from tqdm import tqdm

    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    dist_matrix = torch.empty(num_a, num_b, device=a.device)
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dist_batch = torch.cdist(a_batch, b_batch)
            dist_matrix[i:i+batch_size, j:j+batch_size] = dist_batch.cpu()
    return dist_matrix

def cosine_similarity(a, b):
    if len(a) > 30000:
        return cosine_similarity_batch(a, b, batch_size=30000)

    dot_product = torch.sum(a.unsqueeze(1) * b.unsqueeze(0), dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    norm_prod = norm_a.unsqueeze(1) * norm_b.unsqueeze(0)
    similarity = dot_product / norm_prod

    return similarity


def cosine_similarity_batch(a, b, batch_size=30000):
    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()
    return similarity_matrix


def hungarian_correspondence(similarity_matrix):
    # Convert similarity matrix to a cost matrix by negating the similarity values
    cost_matrix = -similarity_matrix.cpu().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a binary matrix with 1s at matched indices and 0s elsewhere
    num_rows, num_cols = similarity_matrix.shape
    match_matrix = np.zeros((num_rows, num_cols), dtype=int)
    match_matrix[row_indices, col_indices] = 1
    match_matrix = torch.from_numpy(match_matrix).cuda()
    return match_matrix


def gmm(a, b):
    # Compute Gram matrices
    gram_matrix_a = torch.mm(a, a.t())
    gram_matrix_b = torch.mm(b, b.t())

    # Expand dimensions to facilitate broadcasting
    gram_matrix_a = gram_matrix_a.unsqueeze(1)
    gram_matrix_b = gram_matrix_b.unsqueeze(0)

    # Compute Frobenius norm for each pair of vertices using vectorized operations
    correspondence_matrix = torch.norm(gram_matrix_a - gram_matrix_b, p='fro', dim=2)

    return correspondence_matrix

def export_obj(savefile, vertices, faces, uv=None, fuv = None, vnormals=None, fnormals=None):
    with open(savefile, 'w') as f:
        for vi, v in enumerate(vertices):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        if uv is not None:
            for v_uv in uv:
                f.write(f"vt {v_uv[0]} {v_uv[1]} \n")
        if vnormals is not None:
            for vnormal in vnormals:
                f.write(f"vn {vnormal[0]} {vnormal[1]} {vnormal[2]}\n")
        if fuv is not None:
            if fnormals is not None:
                for i in range(len(faces)):
                    face = faces[i]
                    faceuv = fuv[i]
                    fnormal = fnormals[i]
                    f.write(f"f {face[0]+1:d}/{faceuv[0]+1:d}/{fnormal[0]+1:f} {face[1]+1:d}/{faceuv[1]+1:d}/{fnormal[1]+1:f} {face[2]+1:d}/{faceuv[2]+1:d}/{fnormal[2]+1:f}\n")
            else:
                for i in range(len(faces)):
                    face = faces[i]
                    faceuv = fuv[i]
                    f.write(f"f {face[0]+1:d}/{faceuv[0]+1:d} {face[1]+1:d}/{faceuv[1]+1:d} {face[2]+1:d}/{faceuv[2]+1:d}\n")
        elif fnormals is not None:
            for i in range(len(faces)):
                face = faces[i]
                fnormal = fnormals[i]
                f.write(f"f {face[0]+1:d}//{fnormal[0]+1:f} {face[1]+1:d}//{fnormal[1]+1:f} {face[2]+1:d}//{fnormal[2]+1:f}\n")
        else:
            for i in range(len(faces)):
                face = faces[i]
                f.write(f"f {face[0]+1:d} {face[1]+1:d} {face[2]+1:d}\n")