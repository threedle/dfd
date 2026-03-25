### Simplify mesh using fast quadric decimation and save to the same directory
# TODO: include gltf transform simplification in here as well
import os
import argparse
import fast_simplification
import time
import numpy as np

def simplify_from_vs(vpath, fpath, reduction):
    t0 = time.time()

    vertices= np.load(vpath)
    faces = np.load(fpath)

    print(f"Applying simplification with reduction {reduction}...")
    points_out, faces_out, collapses = fast_simplification.simplify(vertices, faces, reduction,
                                                                    return_collapses=True)

    newvpath = os.path.join(os.path.dirname(vpath), f"vertices_simplified{reduction}.npy")
    newfpath = os.path.join(os.path.dirname(fpath), f"faces_simplified{reduction}.npy")
    np.save(newvpath, points_out)
    np.save(newfpath, faces_out)

    t2 = (time.time() - t0) / 60
    print(f"Time taken: {t2} minutes")

    return newvpath, newfpath

def simplify_obj(trimesh, reduction, savename=None):
    t0 = time.time()
    print(f"Loading mesh from {trimesh}...")
    import igl

    if trimesh.endswith(".obj"):
        v, tc, n, f, ftc, _ = igl.readOBJ(trimesh)
        if len(tc) == 0:
            tc = None
            ftc = None
    else:
        v, f = igl.read_triangle_mesh(trimesh)
        tc = None
        ftc = None

    v = v.astype(np.float32)
    f = f.astype(np.int32)

    print(f"Applying simplification with reduction {reduction}...")
    points_out, faces_out, collapses = fast_simplification.simplify(v, f, reduction,
                                                                    return_collapses=True)

    # Transfer texture coordinates if exist
    if tc is not None:
        points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(v, f, collapses)

        # Map tc to lr textures
        reverse_indice_mapping = np.zeros(len(points_out), dtype=int)
        reverse_indice_mapping[indice_mapping] = np.arange(len(indice_mapping))

        # Map v to tc
        vcorners = f.flatten()
        tcorners = ftc.flatten()
        vtotc = np.zeros(len(v), dtype=int)
        vtotc[vcorners] = tcorners

        new_uv = tc[vtotc[reverse_indice_mapping]]
        new_fuv = faces_out

        # Create mapping from hr corners to tc
        hrcorner_to_tc = ftc.flatten() # Based on canonical mapping to corners

        # Create mapping from hr corners to lr points
        hrcorner_to_lrv = indice_mapping[f].flatten() # hr corners => lr points

        # Reverse mapping from lr points to hr corners
        lrv_to_hrcorner = np.zeros(len(points_out), dtype=int)
        lrv_to_hrcorner[hrcorner_to_lrv] = np.arange(len(hrcorner_to_lrv))

        # LR corners to LR points
        lr_corners = faces_out.flatten()

        new_uv = tc[hrcorner_to_tc[lrv_to_hrcorner[lr_corners]]] # F'*3 x 2
        new_fuv = np.arange(len(new_uv)).reshape(-1, 3) # F'*3 x 3


    if savename is None:
        meshname = os.path.basename(trimesh).split(".")[0]
        savename = f"{meshname}_simplified{reduction}.obj"
    savedir = os.path.join(os.path.dirname(trimesh), savename)

    from utils import export_obj

    export_obj(savedir, points_out, faces_out, uv=new_uv if tc is not None else None,
            fuv=new_fuv if tc is not None else None)

    print(f"Mesh saved to {savedir}")
    t2 = (time.time() - t0) / 60
    print(f"Time taken: {t2} minutes")

    return savedir

def simplify_from_file(trimesh, reduction, savename=None):
    t0 = time.time()
    print(f"Loading mesh from {trimesh}...")
    import igl

    if trimesh.endswith(".obj"):
        v, tc, n, f, ftc, _ = igl.readOBJ(trimesh)
        if len(tc) == 0:
            tc = None
            ftc = None
    else:
        v, f = igl.read_triangle_mesh(trimesh)
        tc = None
        ftc = None

    v = v.astype(np.float32)
    f = f.astype(np.int32)

    print(f"Applying simplification with reduction {reduction}...")
    points_out, faces_out, collapses = fast_simplification.simplify(v, f, reduction,
                                                                    return_collapses=True)

    # Transfer texture coordinates if exist
    if tc is not None:
        points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(v, f, collapses)

        # Map tc to lr textures
        reverse_indice_mapping = np.zeros(len(points_out), dtype=int)
        reverse_indice_mapping[indice_mapping] = np.arange(len(indice_mapping))

        # Map v to tc
        vcorners = f.flatten()
        tcorners = ftc.flatten()
        vtotc = np.zeros(len(v), dtype=int)
        vtotc[vcorners] = tcorners

        new_uv = tc[vtotc[reverse_indice_mapping]]
        new_fuv = faces_out

        # Create mapping from hr corners to tc
        hrcorner_to_tc = ftc.flatten() # Based on canonical mapping to corners

        # Create mapping from hr corners to lr points
        hrcorner_to_lrv = indice_mapping[f].flatten() # hr corners => lr points

        # Reverse mapping from lr points to hr corners
        lrv_to_hrcorner = np.zeros(len(points_out), dtype=int)
        lrv_to_hrcorner[hrcorner_to_lrv] = np.arange(len(hrcorner_to_lrv))

        # LR corners to LR points
        lr_corners = faces_out.flatten()

        new_uv = tc[hrcorner_to_tc[lrv_to_hrcorner[lr_corners]]] # F'*3 x 2
        new_fuv = np.arange(len(new_uv)).reshape(-1, 3) # F'*3 x 3


    if savename is None:
        meshname = os.path.basename(trimesh).split(".")[0]
        savename = f"{meshname}_simplified{reduction}.obj"
    savedir = os.path.join(os.path.dirname(trimesh), savename)

    from utils import export_obj

    export_obj(savedir, points_out, faces_out, uv=new_uv if tc is not None else None,
            fuv=new_fuv if tc is not None else None)

    print(f"Mesh saved to {savedir}")
    t2 = (time.time() - t0) / 60
    print(f"Time taken: {t2} minutes")

    return savedir

def simplify(vertices, faces, reduction, savepath, uvs=None, fuv=None):
    t0 = time.time()

    v = vertices.astype(np.float32)
    f = faces.astype(np.int32)
    tc = uvs.astype(np.float32) if uvs is not None else None
    ftc = fuv.astype(np.int32) if fuv is not None else f

    print(f"Applying simplification with reduction {reduction}...")
    points_out, faces_out, collapses = fast_simplification.simplify(v, f, reduction,
                                                                    return_collapses=True)

    # Transfer texture coordinates if exist
    if tc is not None:
        points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(v, f, collapses)

        # Map tc to lr textures
        reverse_indice_mapping = np.zeros(len(points_out), dtype=int)
        reverse_indice_mapping[indice_mapping] = np.arange(len(indice_mapping))

        # Map v to tc
        vcorners = f.flatten()
        tcorners = ftc.flatten()
        vtotc = np.zeros(len(v), dtype=int)
        vtotc[vcorners] = tcorners

        new_uv = tc[vtotc[reverse_indice_mapping]]
        new_fuv = faces_out

        # Create mapping from hr corners to tc
        hrcorner_to_tc = ftc.flatten() # Based on canonical mapping to corners

        # Create mapping from hr corners to lr points
        hrcorner_to_lrv = indice_mapping[f].flatten() # hr corners => lr points

        # Reverse mapping from lr points to hr corners
        lrv_to_hrcorner = np.zeros(len(points_out), dtype=int)
        lrv_to_hrcorner[hrcorner_to_lrv] = np.arange(len(hrcorner_to_lrv))

        # LR corners to LR points
        lr_corners = faces_out.flatten()

        new_uv = tc[hrcorner_to_tc[lrv_to_hrcorner[lr_corners]]] # F'*3 x 2
        new_fuv = np.arange(len(new_uv)).reshape(-1, 3) # F'*3 x 3

    from utils import export_obj

    export_obj(savepath, points_out, faces_out, uv=new_uv if tc is not None else None,
               fuv=new_fuv if tc is not None else None)

    print(f"Mesh saved to {savepath}")
    t2 = (time.time() - t0) / 60
    print(f"Time taken: {t2} minutes")

    return savepath

def simplify_from_gltf_file(gltfpath, reduction, savepath):
    import subprocess

    t0 = time.time()
    subprocess.run(["npx", "@gltf-transform/cli", "weld", gltfpath, savepath])
    subprocess.run(["npx", "@gltf-transform/cli", "simplify", savepath, savepath, "--ratio", str(reduction), "--error", "1"])
    print(f"Mesh saved to {savepath}")
    t2 = (time.time() - t0) / 60
    print(f"Time taken: {t2} minutes")

    return savepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trimesh", type=str, help="Path to the input mesh file")
    parser.add_argument("savepath", type=str, help="Path to save the simplified mesh file")
    parser.add_argument("--reduction", type=float, default=0.5, help="What percentage of edges to collapse.")

    args = parser.parse_args()

    import igl
    v, tc, n, f, ftc, _ = igl.readOBJ(args.trimesh)
    v = v.astype(np.float32)
    f = f.astype(np.int32)
    tc = tc.astype(np.float32) if tc is not None else None
    ftc = ftc.astype(np.int32) if ftc is not None else f

    simplify(v, f, args.reduction, args.savepath, tc, ftc)