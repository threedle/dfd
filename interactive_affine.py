### Self-skinning field: with affine transformations (translation + rotation) ###
import numpy as np
import os
import sys
import torch
import polyscope as ps
import polyscope.imgui as psim
import argparse
import datetime
import dill as pickle
import copy
from utils import get_rotation_x, get_rotation_y, get_rotation_z, build_affine_matrix
from GLOBALS import sourcecolor, vectorcolor, targetcolor, constraintcolor, defaultcolor, pointradius, curveradius

if __name__ == "__main__":
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objdir', type=str, help='path to obj file')
    parser.add_argument('--weightsdir', type=str, help='path to saved MLP weights (image features distilled)')
    parser.add_argument('--featurepath', type=str, help='path to saved vertex features (if not using MLP). Symmetry evaluation will NOT be available.', default=None)
    parser.add_argument('--argsdir', type=str, help='path to parameters json. defaults to weightsdir.replace(".pth", ".json")', default=None)
    parser.add_argument('--savedir', type=str, help='directory to save exports to', default=None)
    parser.add_argument('--texturedir', type=str, help='path to texture image', default=None)
    parser.add_argument('--maxv', type=int, default=30000, help="meshes with # v over this don't precompute weights")

    args = parser.parse_args()

    from pathlib import Path
    savedir = args.savedir
    if savedir is None:
        savedir = os.path.join(os.path.dirname(args.weightsdir), "exports")
    Path(savedir).mkdir(parents=True, exist_ok=True)

    import igl
    textureimg = None
    fuv = None
    if args.objdir.endswith(".obj"):
        v, vt, n, f, ftc, _ = igl.readOBJ(args.objdir)
        if len(vt) == 0:
            fuv = None
        else:
            # Convert to corner UVs
            if ftc is not None and len(ftc) > 0:
                fuv = vt[ftc].reshape(-1, 2)
            else:
                fuv = vt[f].reshape(-1, 2)

    elif args.objdir.endswith(".glb"):
        import trimesh
        mesh = trimesh.load(args.objdir)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        v = mesh.vertices
        f = mesh.faces

        material = None
        fuv = None
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material

            if isinstance(material, trimesh.visual.material.MultiMaterial):
                material = material.materials[0]

            if isinstance(material, trimesh.visual.material.SimpleMaterial):
                material = material.to_pbr()

            if isinstance(material, trimesh.visual.material.PBRMaterial):
                textureimg = material.baseColorTexture
                textureimg = np.array(textureimg)[:,:,:3]
                textureimg = textureimg / 255.0
            else:
                print(f"Trimesh material unsupported: {type(material)}")

        if hasattr(mesh.visual, 'uv') and material is not None:
            uvs = mesh.visual.uv
            fuv = uvs[f].reshape(-1, 2)
    else:
        v, f = igl.read_triangle_mesh(args.objdir)
        fuv = None

    vertices = np.array(v, dtype=np.float32)
    faces = np.array(f, dtype=np.int32)

    from utils import cube_normalize
    vertices = cube_normalize(vertices)

    # Vertices with homogeneous coordinates
    vertices_hom = torch.from_numpy(np.hstack((vertices, np.ones((vertices.shape[0], 1), dtype=np.float32))))

    featurepath = args.featurepath
    weightsdir = args.weightsdir
    if featurepath is None and weightsdir is None:
        raise ValueError("Either --featurepath or --weightsdir must be provided")

    if weightsdir is None:
        featuremlp = None
        meshfeatures = torch.load(featurepath, map_location='cpu', weights_only=True)
        print(f"Loaded features from {featurepath}")
    else:
        from featuremlp import FeatureMLP
        import json

        argsdir = args.argsdir
        if argsdir is None:
            argsdir = args.weightsdir.replace(".pth", ".json")

        with open(argsdir, 'r') as f:
            argsdict = json.load(f)
        argsdict = argparse.Namespace(**argsdict)
        featuremlp = FeatureMLP(argsdict.nlayers, argsdict.width, out_dim=argsdict.out_dim, positional_encoding=argsdict.positional_encoding,
                    sigma=argsdict.sigma)

        featuremlp.load_state_dict(torch.load(args.weightsdir, map_location='cpu', weights_only=True))
        featuremlp.eval()

        # Generate mesh features
        with torch.no_grad():
            meshfeatures = featuremlp(torch.from_numpy(vertices).float())

    norms = torch.linalg.norm(meshfeatures, dim=1)

    distances = None
    geodesic_solver = None
    # Initialize cache so we only have to solve once per vertex
    geodesics_cache = {}
    geopower = 0

    # Get geodesic or euclidean distances
    import potpourri3d as pp3d

    # NOTE: This OOMs for large meshes and no easy way to fix it
    # = Stateful solves (much faster if computing distance many times)
    geodesic_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

    print(f"Done with geodesic solver")

    # NOTE: for large model, we compute individual vertex weights on the fly
    from utils import cdist
    largev = len(meshfeatures) > args.maxv
    selfweights_path = os.path.join(savedir, "self_weights.pt")
    self_weights_handle = None
    self_weights_geo = None

    if os.path.exists(selfweights_path):
        print("Loading self weights...")
        self_weights = torch.load(selfweights_path, weights_only=True)
        self_weights_anchored = self_weights.clone().detach()
        print(f"Loaded self weights: {self_weights.shape}")
    elif not largev:
        self_weights = cdist(meshfeatures, meshfeatures) # V x V matrix of distances
        self_weights = 1 - self_weights

        # Clamp to 0-1
        self_weights = torch.clamp(self_weights, 0, 1)
        self_weights_anchored = self_weights.clone().detach()
        print(f"Computed self weights: {self_weights.shape}")
        torch.save(self_weights, selfweights_path)
    else:
        self_weights = None
        self_weights_anchored = None

    def compute_handle_weights_onthefly(vi, anchor_list):
        """Compute handle weights for vertex vi without the full V x V matrix.

        Computes only the vi-th row of the similarity matrix and subtracts
        the per-anchor max similarity, equivalent to self_weights_anchored[vi]
        when self_weights is precomputed.
        """
        handle_weights = 1 - cdist(meshfeatures[[vi]], meshfeatures).squeeze(0)
        handle_weights = torch.clamp(handle_weights, 0, 1)
        if len(anchor_list) > 0:
            anchor_sims = 1 - cdist(meshfeatures[anchor_list], meshfeatures)
            anchor_sims = torch.clamp(anchor_sims, 0, 1)
            handle_weights = handle_weights - torch.max(anchor_sims, dim=0)[0]
            handle_weights = torch.clamp(handle_weights, 0, 1)
        return handle_weights

    symmetry_labels = ["None"]
    symmetries = []
    symmetry_groups = []

    if featuremlp is not None:
        # Check if weights are symmetric across one of the major axes
        # Reflect vertices along the x, y, and z axes, compute weights, and check how similar they are
        for i in range(3): # NOTE: checks for reflection across YZ plane, XZ plane, XY plane respectively
            # Find all vertices that are on same side of axis as vertex
            symmetry_group0 = np.where(vertices[:, i] >= 0)[0]
            symmetry_group1 = np.where(vertices[:, i] < 0)[0]

            reflected_vertices = vertices[symmetry_group0].copy()
            reflected_vertices[:, i] *= -1
            with torch.no_grad():
                reflected_dists = torch.linalg.norm(meshfeatures[symmetry_group0] - featuremlp(torch.from_numpy(reflected_vertices).float()), dim=1)

            # Check if features are similar
            weight_diff = torch.mean(reflected_dists)
            print(f"Weight difference across axis {i}: {weight_diff.item()}")

            if weight_diff < 0.1:  # Adjust threshold as needed
                symmetry_axis = i
                print(f"Found symmetry across axis {i}")
                symmetries.append(symmetry_axis)
                symmetry_groups.append((symmetry_group0, symmetry_group1))

                if i == 0:
                    symmetry_labels.append("YZ Plane")
                elif i == 1:
                    symmetry_labels.append("XZ Plane")
                elif i == 2:
                    symmetry_labels.append("XY Plane")

    # Get matplotlib colormap
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    ######## Deformations ########
    last_action = None
    vertices_def_cache = [] # Contains sequence of deformations in the order they were applied
    vertices_def = np.copy(vertices) # Deformed vertices
    last_vertices_def = np.copy(vertices) # Last vertices def
    precompute_vertices = np.copy(vertices) # Global vertices precompute
    vertices_def_hom = torch.from_numpy(np.hstack((vertices_def, np.ones((vertices_def.shape[0], 1), dtype=np.float32)))) # V x 4
    vertex_index = None
    iterative_control_points = [] # Stores history of iterative control points for visualization

    # Symmetry
    symmetry = "None"
    symmetry_axis = None
    show_symmetry_plane = True
    symmetry_group0, symmetry_group1 = None, None

    # Mode
    mode = "iterative" # Can be iterative or global
    modes = ['iterative']
    control_points = {}
    cp_def_cache = {} # Dictionary of deformation values per control point
    showinfluence = True
    geoweights = []

    # Translations for determining deformation
    x_translation = 0
    y_translation = 0
    z_translation = 0

    # Degrees for the rotation matrices
    x_rotation = 0
    y_rotation = 0
    z_rotation = 0

    # Scaling
    x_scale = 1.0
    y_scale = 1.0
    z_scale = 1.0

    # Identity affine for lerp
    deformation_state = None
    identity_affine = torch.eye(4, dtype=torch.float32)
    affine_handle = None
    affine_matrix = None

    anchors = []
    add_anchors = False
    ps_anchors = None

    compute_anchors = []
    currentanchor = None

    # Define global colors
    anchorbase = (0.988, 0.275, 0.667)
    anchorselect = (1, 0, 0)

    cpbase = (1, 1, 0)  # Yellow
    cpselect = (0, 1, 0)  # Green

    def callback():
        # If we want to use local variables & assign to them in the UI code below,
        # we need to mark them as nonlocal. This is because of how Python scoping
        # rules work, not anything particular about Polyscope or ImGui.
        # Of course, you can also use any other kind of python variable as a controllable
        # value in the UI, such as a value from a dictionary, or a class member. Just be
        # sure to assign the result of the ImGui call to the value, as in the examples below.
        #
        # If these variables are defined at the top level of a Python script file (i.e., not
        # inside any method), you will need to use the `global` keyword instead of `nonlocal`.

        global args, ps_mesh, ps_selection, vertices, faces, vertex_index, vertices_hom, vertices_def_hom
        global geoweights, precompute_vertices, iterative_control_points
        global vertices_def_cache, vertices_def, last_vertices_def, vertices, cp_def_cache
        global symmetries, symmetry_labels, symmetry_groups, symmetry, symmetry_axis, show_symmetry_plane
        global symmetry_group0, symmetry_group1
        global x_translation, y_translation, z_translation, geopower
        global x_rotation, y_rotation, z_rotation
        global x_scale, y_scale, z_scale, mode, modes, control_points
        global identity_affine, affine_handle, affine_matrix, deformation_state
        global showinfluence, last_action
        global anchors, add_anchors, ps_anchors, ps_plane, compute_anchors
        global textureimg, fuv
        global anchorbase, anchorselect, cpbase, cpselect

        global largev, self_weights_geo
        global meshfeatures, self_weights, self_weights_anchored, self_weights_handle

        global geodesic_solver, geodesics_cache
        global distances

        vrange = np.arange(len(vertices))
        frange = np.arange(len(vertices), len(vertices) + len(faces))

        # == Settings
        # Use settings like this to change the UI appearance.
        # Note that it is a push/pop pair, with the matching pop() below.
        psim.PushItemWidth(150)

        # == Title window
        psim.TextUnformatted("Interactive Deformations")
        psim.Separator()

        # == Reset button ==
        if(psim.Button("Reset")):
            ## Reset the selections ##
            ps.reset_selection()

            last_action = None
            ps_selection = ps.register_point_cloud("selection", np.zeros((1, 3)), radius=pointradius, color=(0, 1, 0), enabled=False)
            ps_selection.set_radius(pointradius, relative=False)
            ps_selection.set_ignore_slice_plane(ps_plane, True)

            vertex_index = None
            iterative_control_points = []

            x_translation = 0
            y_translation = 0
            z_translation = 0

            x_rotation = 0
            y_rotation = 0
            z_rotation = 0

            x_scale = 1.0
            y_scale = 1.0
            z_scale = 1.0

            show_symmetry_plane = True
            symmetry = "None"
            symmetry_group0, symmetry_group1 = None, None
            if symmetry_axis is None:
                ps_plane.set_draw_plane(False)
                ps_plane.set_active(False)
            else:
                if symmetry_axis == 0:
                    ps_plane.set_pose((0., 0., 0.), (1., 0., 0.))
                elif symmetry_axis == 1:
                    ps_plane.set_pose((0., 0., 0.), (0., 1., 0.))
                elif symmetry_axis == 2:
                    ps_plane.set_pose((0., 0., 0.), (0., 0., 1.))
                ps_plane.set_draw_plane(True)
                ps_plane.set_active(True)

            affine_handle = None
            affine_matrix = None
            deformation_state = None

            anchors = []
            add_anchors = False

            # Clear the anchor point cloud
            if ps_anchors is not None:
                ps_anchors.set_enabled(False)

            # Compute anchors
            compute_anchors = []

            if not largev:
                self_weights_anchored = self_weights.clone().detach()
            self_weights_handle = None
            self_weights_geo = None

            ## Reset the mesh and deformation quantities ##
            cp_def_cache = {}
            vertices_def_cache = []
            last_vertices_def = np.copy(vertices) # Reset last vertices def
            vertices_def = np.copy(vertices) # Reset deformed vertices
            precompute_vertices = np.copy(vertices) # Reset precompute vertices
            vertices_def_hom = torch.from_numpy(np.hstack((vertices_def, np.ones((vertices_def.shape[0], 1), dtype=np.float32)))) # V x 4

            ps_mesh.update_vertex_positions(vertices)
            ps_mesh.remove_all_quantities()

            if textureimg is not None:
                ps_mesh.add_parameterization_quantity("uv", fuv, defined_on='corners', enabled=True)
                ps_mesh.add_color_quantity("texture", textureimg, defined_on='texture', enabled=True, param_name='uv')

        psim.SameLine()

        # NOTE: Undo most recent vertices_def_cache entry
        if(psim.Button("Undo")):
            if last_action == "undo":
                return

            vertices_def = last_vertices_def.copy()
            ps_mesh.update_vertex_positions(vertices_def)

            if len(iterative_control_points) > 0:
                iterative_control_points.pop()

                if len(iterative_control_points) > 0:
                    ps_selection = ps.register_point_cloud("selection", vertices_def[iterative_control_points], radius=pointradius, enabled=True)
                    ps_selection.set_radius(pointradius, relative=False)
                else:
                    ps_selection.set_enabled(False)

            if len(vertices_def_cache) > 0:
                vertices_def_cache.pop()

            vertex_index = None
            ps.reset_selection()

            last_action = "undo"

            if last_action == "add_anchor":
                anchors.pop()

                if len(anchors) > 0:
                    ps_anchors = ps.register_point_cloud("anchors", vertices_def[anchors], radius=pointradius, color=constraintcolor, enabled=True)
                    ps_anchors.set_radius(pointradius, relative=False)

                    if not largev:
                        self_weights_anchored = self_weights - torch.max(self_weights[anchors, :], dim=0)[0]
                        self_weights_anchored = torch.clamp(self_weights_anchored, 0, 1)
                else:
                    if not largev:
                        self_weights_anchored = self_weights.clone().detach()

        psim.SameLine()

        if(psim.Button("Export")):
            datefolder = datetime.datetime.now().strftime("%d-%m-%Y__%H:%M:%S")
            exportdir = os.path.join(savedir, datefolder)
            Path(exportdir).mkdir(parents=True, exist_ok=True)

            # Save the deformed mesh vertices
            np.save(os.path.join(exportdir, "deformed_vertices.npy"), vertices_def)

            # Save deformation values as pickled list
            with open(os.path.join(exportdir, "deformation_values.pkl"), 'wb') as f:
                pickle.dump(vertices_def_cache, f)

            # Save the pulled vertex positions as txt
            with open(os.path.join(exportdir, "control_points.txt"), 'w') as f:
                for cachevalues in vertices_def_cache:
                    vi = cachevalues[0]

                    if isinstance(vi, list):
                        for v in vi:
                            vposition = vertices_def[v]
                            f.write(f"{v} {vposition[0]} {vposition[1]} {vposition[2]}\n")
                    else:
                        vposition = vertices_def[vi]
                        f.write(f"{vi} {vposition[0]} {vposition[1]} {vposition[2]}\n")

            # Save the anchors as constraints
            if len(anchors) > 0:
                with open(os.path.join(exportdir, "anchor_points.txt"), 'w') as f:
                    for vi in anchors:
                        vposition = vertices[vi]
                        f.write(f"{vi} {vposition[0]} {vposition[1]} {vposition[2]}\n")

        psim.Separator()

        # Symmetry list: if symmetries are found
        if len(symmetries) > 0:
            symmetrychange = psim.BeginCombo("Symmetry Plane", symmetry)
            if symmetrychange:
                for val in symmetry_labels:
                    _, selected = psim.Selectable(val, symmetry==val)
                    if selected:
                        symmetry = val

                        if symmetry == "None":
                            symmetry_axis = None
                            symmetry_group0, symmetry_group1 = None, None
                        elif symmetry == "YZ Plane":
                            symmetry_axis = 0
                            symmetry_group0, symmetry_group1 = symmetry_groups[symmetry_labels.index(symmetry) - 1]
                        elif symmetry == "XZ Plane":
                            symmetry_axis = 1
                            symmetry_group0, symmetry_group1 = symmetry_groups[symmetry_labels.index(symmetry) - 1]
                        elif symmetry == "XY Plane":
                            symmetry_axis = 2
                            symmetry_group0, symmetry_group1 = symmetry_groups[symmetry_labels.index(symmetry) - 1]

                        if show_symmetry_plane:
                            # Show the symmetry plane
                            if symmetry_axis is None:
                                ps_plane.set_draw_plane(False)
                                ps_plane.set_active(False)
                            else:
                                if symmetry_axis == 0:
                                    ps_plane.set_pose((0., 0., 0.), (1., 0., 0.))
                                elif symmetry_axis == 1:
                                    ps_plane.set_pose((0., 0., 0.), (0., 1., 0.))
                                elif symmetry_axis == 2:
                                    ps_plane.set_pose((0., 0., 0.), (0., 0., 1.))
                                ps_plane.set_draw_plane(True)
                                ps_plane.set_active(True)
                psim.EndCombo()

        # Toggle to show the symmetry plane
        if symmetry_axis is not None:
            psim.SameLine()
            showsymchange, show_symmetry_plane = psim.Checkbox("Show Symmetry Plane", show_symmetry_plane)

            if showsymchange:
                if show_symmetry_plane:
                    # Show the symmetry plane
                    if symmetry_axis == 0:
                        ps_plane.set_pose((0., 0., 0.), (1., 0., 0.))
                    elif symmetry_axis == 1:
                        ps_plane.set_pose((0., 0., 0.), (0., 1., 0.))
                    elif symmetry_axis == 2:
                        ps_plane.set_pose((0., 0., 0.), (0., 0., 1.))
                    ps_plane.set_draw_plane(True)
                    ps_plane.set_active(True)
                else:
                    ps_plane.set_draw_plane(False)
                    ps_plane.set_active(False)

        ### Show influence ###
        influencechange, showinfluence = psim.Checkbox("Show Influence", showinfluence)
        if influencechange:
            if not showinfluence:
                # Reset the mesh colors to either default or the texture
                ps_mesh.remove_quantity("influence")
                if textureimg is not None:
                    ps_mesh.add_parameterization_quantity("uv", fuv, defined_on='corners', enabled=True)
                    ps_mesh.add_color_quantity("texture", textureimg, defined_on='texture', enabled=True, param_name='uv')
            else:
                # If vertex index is set, update the influence colors
                if vertex_index is not None:
                    # Set color based on influence
                    influence = self_weights_handle.detach().cpu().numpy()

                    colors = sm.to_rgba(influence)[:, :3]

                    # Set self vertex to yellow
                    colors[vertex_index] = (1, 1, 0)

                    ps_mesh.add_color_quantity("influence", colors, defined_on='vertices', enabled=True)

        # == On click event, highlight the selected curve/control points ==
        ### Keep track of the clicked curves/control points
        # NOTE: Structure gives you the string name of the structure
        selection = ps.get_selection()
        structure = selection.structure_name
        index = selection.local_index
        structure_data = selection.structure_data

        if structure == "mesh" and index < len(vertices):
            if not add_anchors:
                # Checks if selection is new vertex or first click
                if (vertex_index is not None and index != vertex_index) or vertex_index is None:
                    vertex_index = index

                    if len(iterative_control_points) > 0 and vertex_index not in iterative_control_points:
                        ps_selection = ps.register_point_cloud("selection", vertices_def[iterative_control_points + [vertex_index]], radius=pointradius, enabled=True)
                        ps_selection.set_radius(pointradius, relative=False)
                        select_idx = -1
                    elif len(iterative_control_points) == 0:
                        ps_selection = ps.register_point_cloud("selection", vertices_def[[vertex_index]], radius=pointradius, enabled=True)
                        ps_selection.set_radius(pointradius, relative=False)
                        select_idx = -1
                    else: # already in iterative control points
                        ps_selection = ps.register_point_cloud("selection", vertices_def[iterative_control_points], radius=pointradius, enabled=True)
                        ps_selection.set_radius(pointradius, relative=False)
                        select_idx = iterative_control_points.index(vertex_index)

                    # Colors: yellow for iterative cps, green for current
                    colors = np.zeros((len(iterative_control_points)+1, 3), dtype=np.float32)
                    colors[:] = (1, 1, 0)
                    colors[select_idx] = (0, 1, 0)
                    ps_selection.add_color_quantity("selection", colors, enabled=True)

                    # Slice plane
                    ps_selection.set_ignore_slice_plane(ps_plane, True)

                    # Every new click is a new deformation
                    x_translation, y_translation, z_translation = 0, 0, 0
                    x_rotation, y_rotation, z_rotation = 0, 0, 0
                    x_scale, y_scale, z_scale = 1.0, 1.0, 1.0

                    ## Handle-specific weights
                    if largev:
                        self_weights_handle = compute_handle_weights_onthefly(vertex_index, anchors)
                    else:
                        self_weights_handle = self_weights_anchored[vertex_index].clone()

                    # Geodesic distance adjustment if applicable
                    if geopower > 0:
                        if vertex_index not in geodesics_cache:
                            geodistance = geodesic_solver.compute_distance(vertex_index)

                            # Replace nan values with max distance
                            geodistance[np.isnan(geodistance)] = np.nanmax(geodistance)

                            # HACK: Normalize to [pointradius, 0.98] so the influence still scales appropriately
                            geodistance = (geodistance - np.min(geodistance)) / (np.max(geodistance) - np.min(geodistance))
                            geodistance = 0.99 * geodistance

                            geodesics_cache[vertex_index] = torch.from_numpy(geodistance).float()

                        self_weights_geo = self_weights_handle * (1 - geodesics_cache[vertex_index]) ** geopower

                        # Rescale the influence to 1 for the self influence
                        if self_weights_geo[vertex_index] > 0:
                            self_weights_geo = self_weights_geo / self_weights_geo[vertex_index]
                            self_weights_geo = torch.clamp(self_weights_geo, 0, 1)
                            assert self_weights_geo.min() >= 0, f"Min: {self_weights_geo.min()}"
                            assert self_weights_geo.max() == 1, f"Max: {self_weights_geo.max()}"
                    else:
                        self_weights_geo = self_weights_handle.clone()

                    # Update affine handle
                    affine_handle = (1 - self_weights_geo[:, None, None]) * identity_affine[None].repeat(len(vertices), 1, 1)

                    if showinfluence:
                        # Set color based on influence
                        influence = self_weights_geo.detach().cpu().numpy()

                        colors = sm.to_rgba(influence)[:, :3]

                        # Set self vertex to yellow
                        colors[vertex_index] = (1, 1, 0)

                        ps_mesh.add_color_quantity("influence", colors, defined_on='vertices', enabled=True)

                    # NOTE: This is where we set in stone the previous deformation state if any real changes have been made
                    if affine_matrix is not None:
                        last_vertices_def = np.copy(vertices_def) # Update last vertices def
                        vertices_def_hom = torch.from_numpy(np.hstack((vertices_def, np.ones((vertices_def.shape[0], 1), dtype=np.float32)))) # V x 4
                        vertices_def_cache.append(deformation_state)

                        iterative_control_points.append(deformation_state[0])

                        ps_selection = ps.register_point_cloud("selection", vertices_def[iterative_control_points + [vertex_index]], radius=pointradius, enabled=True)
                        ps_selection.set_radius(pointradius, relative=False)

                    # Initialize new state
                    affine_matrix = None
                    deformation_state = (vertex_index, copy.deepcopy(anchors), self_weights_geo.numpy(), identity_affine.numpy(), None, symmetry_axis)
                    last_action = "add_control_point"
            else:
                # Checks: selection is a new vertex and not already an anchor and not largev
                if index != vertex_index and index not in anchors and not largev:
                    anchors.append(index)

                    # Update anchor point cloud
                    ps_anchors = ps.register_point_cloud("anchors", vertices_def[anchors], radius=pointradius, color=constraintcolor, enabled=True)
                    ps_anchors.set_radius(pointradius, relative=False)

                    self_weights_anchored = self_weights - torch.max(self_weights[anchors, :], dim=0)[0]
                    self_weights_anchored = torch.clamp(self_weights_anchored, 0, 1)

                    # Update weights if vertex index is set (otherwise they will be updated next time a selection is made)
                    if vertex_index is not None:
                        self_weights_handle = self_weights_anchored[vertex_index].clone()

                        # Geodesic distance adjustment if applicable
                        if geopower > 0:
                            if vertex_index not in geodesics_cache:
                                geodistance = geodesic_solver.compute_distance(vertex_index)

                                # Replace nan values with max distance
                                geodistance[np.isnan(geodistance)] = np.nanmax(geodistance)

                                # HACK: Normalize to [0, 0.99] so the influence still scales appropriately
                                geodistance = (geodistance - np.min(geodistance)) / (np.max(geodistance) - np.min(geodistance))
                                geodistance = 0.99 * geodistance

                                geodesics_cache[vertex_index] = torch.from_numpy(geodistance).float()

                            self_weights_geo = self_weights_handle * (1 - geodesics_cache[vertex_index]) ** geopower

                            # Rescale the influence to 1 for the self influence
                            # NOTE: Only do this if the vertex influence is not close to 1
                            if self_weights_geo[vertex_index] > 0:
                                self_weights_geo = self_weights_geo / self_weights_geo[vertex_index]
                                self_weights_geo = torch.clamp(self_weights_geo, 0, 1)
                                assert self_weights_geo.min() >= 0, f"Min: {self_weights_geo.min()}"
                                assert self_weights_geo.max() == 1, f"Max: {self_weights_geo.max()}"

                        else:
                            self_weights_geo = self_weights_handle.clone()

                        # Update affine handle
                        affine_handle = (1 - self_weights_geo[:, None, None]) * identity_affine[None].repeat(len(vertices), 1, 1)

                        # Update the influence based on the new anchors
                        if showinfluence:
                            # Set color based on influence
                            influence = self_weights_geo.detach().cpu().numpy()
                            colors = sm.to_rgba(influence)[:, :3]

                            # Set self vertex to yellow
                            colors[vertex_index] = (1, 1, 0)

                            ps_mesh.add_color_quantity("influence", colors, defined_on='vertices', enabled=True)

                        # Dynamically update the mesh deformation
                        if affine_matrix is not None:
                            affine_all = affine_matrix[None].repeat(len(vertices), 1, 1) # V x 4 x 4

                            # Use the weights to lerp between identity and the affine transformation
                            affine_weighted = affine_handle + self_weights_geo[:, None, None] * affine_all # V x 4 x 4

                            last_vertices_def = np.copy(vertices_def) # Update last vertices def
                            vertices_def = torch.bmm(affine_weighted, vertices_def_hom[:, :, None]).numpy()[:, :3, 0] # V x 3
                            ps_mesh.update_vertex_positions(vertices_def)

                            # Update selection position
                            ps_selection.update_point_positions(vertices_def[iterative_control_points + [vertex_index]])

                            # Update cache
                            deformation_state = (vertex_index, copy.deepcopy(anchors), self_weights_geo.numpy(), identity_affine.numpy(), affine_matrix.numpy(), symmetry_axis)

                    last_action = "add_anchor"

        # Reset the selection if not a vertex and we're not adding anchors
        if structure != "mesh" and not add_anchors:
            if affine_matrix is not None:
                last_vertices_def = np.copy(vertices_def) # Update last vertices def
                vertices_def_hom = torch.from_numpy(np.hstack((vertices_def, np.ones((vertices_def.shape[0], 1), dtype=np.float32)))) # V x 4
                vertices_def_cache.append(deformation_state)

                iterative_control_points.append(deformation_state[0])
                ps_selection = ps.register_point_cloud("selection", vertices_def[iterative_control_points], radius=pointradius, enabled=True)
                ps_selection.set_radius(pointradius, relative=False)

                deformation_state = None

            vertex_index = None
            affine_matrix = None
            ps_selection.set_enabled(False)
            x_translation, y_translation, z_translation = 0, 0, 0
            x_rotation, y_rotation, z_rotation = 0, 0, 0
            x_scale, y_scale, z_scale = 1.0, 1.0, 1.0

            ps_mesh.remove_quantity("influence")

        # Latent anchoring
        anchorchange, add_anchors = psim.Checkbox("Add anchors", add_anchors)

        geopower_change, geopower = psim.SliderFloat("Geodesic Power", geopower, v_min=0, v_max=10)
        if geopower_change and vertex_index is not None:
            if vertex_index not in geodesics_cache:
                geodistance = geodesic_solver.compute_distance(vertex_index)
                geodistance[np.isnan(geodistance)] = np.nanmax(geodistance)

                # HACK: Normalize to [0, 0.99] so the influence still scales appropriately
                geodistance = (geodistance - np.min(geodistance)) / (np.max(geodistance) - np.min(geodistance))
                geodistance = 0.99 * geodistance

                geodesics_cache[vertex_index] = torch.from_numpy(geodistance).float()

            # Geodesic distance adjustment if applicable
            if geopower > 0:
                self_weights_geo = self_weights_handle * (1 - geodesics_cache[vertex_index]) ** geopower

            # Rescale the influence to 1 for the self influence
            if self_weights_geo[vertex_index] > 0:
                self_weights_geo = self_weights_geo / self_weights_geo[vertex_index]
                self_weights_geo = torch.clamp(self_weights_geo, 0, 1)
                assert self_weights_geo.min() >= 0, f"Min: {self_weights_geo.min()}"
                assert self_weights_geo.max() == 1, f"Max: {self_weights_geo.max()}"

            # Update affine handle
            affine_handle = (1 - self_weights_geo[:, None, None]) * identity_affine[None].repeat(len(vertices), 1, 1)

            if showinfluence:
                influence = self_weights_geo.detach().cpu().numpy()

                colors = sm.to_rgba(influence)[:, :3]

                # Set self vertex to yellow
                colors[vertex_index] = (1, 1, 0)

                ps_mesh.add_color_quantity("influence", colors, defined_on='vertices', enabled=True)

            # Dynamically update the mesh deformation
            if affine_matrix is not None and len(iterative_control_points) > 0:
                affine_all = affine_matrix[None].repeat(len(vertices), 1, 1) # V x 4 x 4

                # Use the weights to lerp between identity and the affine transformation
                affine_weighted = affine_handle + self_weights_geo[:, None, None] * affine_all # V x 4 x 4

                last_vertices_def = np.copy(vertices_def) # Update last vertices def
                vertices_def = torch.bmm(affine_weighted, vertices_def_hom[:, :, None]).numpy()[:, :3, 0] # V x 3
                ps_mesh.update_vertex_positions(vertices_def)

                # Update selection position
                ps_selection.update_point_positions(vertices_def[iterative_control_points + [vertex_index]])

                # Update cache
                deformation_state = (vertex_index, copy.deepcopy(anchors), self_weights_geo.numpy(), identity_affine.numpy(), affine_matrix.numpy(), symmetry_axis)

        psim.Separator()

        #### Translation and Rotation ####
        # NOTE: In standard affine transformations, the translation is applied after the rotation.
        psim.TextUnformatted("Translation")
        x_change, x_translation = psim.SliderFloat("Xt", x_translation, v_min=-1, v_max=1)
        y_change, y_translation = psim.SliderFloat("Yt", y_translation, v_min=-1, v_max=1)
        z_change, z_translation = psim.SliderFloat("Zt", z_translation, v_min=-1, v_max=1)

        psim.Separator()

        psim.TextUnformatted("Rotation")
        x_rotation_change, x_rotation = psim.SliderFloat("Xr", x_rotation, v_min=-90, v_max=90)
        y_rotation_change, y_rotation = psim.SliderFloat("Yr", y_rotation, v_min=-90, v_max=90)
        z_rotation_change, z_rotation = psim.SliderFloat("Zr", z_rotation, v_min=-90, v_max=90)

        psim.Separator()

        psim.TextUnformatted("Scale")
        x_scale_change, x_scale = psim.SliderFloat("Xs", x_scale, v_min=0.1, v_max=3.0)
        y_scale_change, y_scale = psim.SliderFloat("Ys", y_scale, v_min=0.1, v_max=3.0)
        z_scale_change, z_scale = psim.SliderFloat("Zs", z_scale, v_min=0.1, v_max=3.0)

        psim.Separator()

        if (x_change or y_change or z_change or x_rotation_change or y_rotation_change or z_rotation_change \
            or x_scale_change or y_scale_change or z_scale_change) and \
                (vertex_index is not None):

            # Build the affine transformation matrix
            translation = torch.tensor([x_translation, y_translation, z_translation])

            x_theta = torch.deg2rad(torch.tensor(x_rotation))
            y_theta = torch.deg2rad(torch.tensor(y_rotation))
            z_theta = torch.deg2rad(torch.tensor(z_rotation))

            x_rotation_matrix = get_rotation_x(x_theta)
            y_rotation_matrix = get_rotation_y(y_theta)
            z_rotation_matrix = get_rotation_z(z_theta)

            rotation_matrix = z_rotation_matrix @ y_rotation_matrix @ x_rotation_matrix

            scale = torch.tensor([x_scale, y_scale, z_scale])

            affine_matrix = build_affine_matrix(rotation_matrix, translation, scale)

            # Duplicate across all vertices
            affine_all = affine_matrix[None].repeat(len(vertices), 1, 1) # V x 4 x 4

            # NOTE: To apply symmetry to the rotation, we reverse the angle corresponding to the symmetry axis
            if symmetry_axis is not None and len(symmetries) > 0:

                if vertices[vertex_index, symmetry_axis] >= 0:
                    reflected_group = symmetry_group1
                else:
                    reflected_group = symmetry_group0

                if symmetry_axis == 0:
                    # Rotations about the x-axis are unaffected
                    x_rotation_matrix_reflected = x_rotation_matrix
                    # HACK: try reflecting the x rotation as well
                    # x_rotation_matrix_reflected = get_rotation_x(-x_theta)
                    y_rotation_matrix_reflected = get_rotation_y(-y_theta)
                    z_rotation_matrix_reflected = get_rotation_z(-z_theta)
                elif symmetry_axis == 1:
                    # Rotations about the y-axis are unaffected
                    x_rotation_matrix_reflected = get_rotation_x(-x_theta)
                    y_rotation_matrix_reflected = y_rotation_matrix
                    z_rotation_matrix_reflected = get_rotation_z(-z_theta)
                else:
                    # Rotations about the z-axis are unaffected
                    x_rotation_matrix_reflected = get_rotation_x(-x_theta)
                    y_rotation_matrix_reflected = get_rotation_y(-y_theta)
                    z_rotation_matrix_reflected = z_rotation_matrix

                reflected_rotation_matrix = z_rotation_matrix_reflected @ y_rotation_matrix_reflected @ x_rotation_matrix_reflected
                reflected_translation = translation.clone()
                reflected_translation[symmetry_axis] *= -1

                reflected_affine = build_affine_matrix(reflected_rotation_matrix, reflected_translation, scale)

                affine_all[reflected_group] = reflected_affine

            # Use the weights to lerp between identity and the affine transformation
            affine_weighted = affine_handle + self_weights_geo[:, None, None] * affine_all # V x 4 x 4

            vertices_def = torch.bmm(affine_weighted, vertices_def_hom[:, :, None]).numpy()[:, :3, 0] # V x 3
            ps_mesh.update_vertex_positions(vertices_def)

            # Update selection position
            ps_selection.update_point_positions(vertices_def[iterative_control_points + [vertex_index]])

            # Update cache
            deformation_state = (vertex_index, copy.deepcopy(anchors), self_weights_geo.numpy(), identity_affine.numpy(), affine_matrix.numpy(), symmetry_axis)

        psim.PopItemWidth()

    ps.init()
    ps.remove_all_structures()

    # min_y_height = np.min(vertices[:, 1])
    # ps.set_ground_plane_height(min_y_height)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_darkness(0.5)

    ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=None, color=defaultcolor)
    ps_mesh.set_selection_mode("vertices_only")

    if args.texturedir is not None:
        # Load texture
        from PIL import Image
        textureimg = Image.open(args.texturedir).convert('RGB')
        textureimg = np.array(textureimg)
        textureimg = textureimg / 255.0

    if fuv is not None and textureimg is not None:
        ps_mesh.add_parameterization_quantity("uv", fuv, defined_on='corners', enabled=True)
        ps_mesh.add_color_quantity("texture", textureimg, defined_on='texture', enabled=True, param_name='uv')

    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_active(False)

    ps_mesh.set_ignore_slice_plane(ps_plane, True)

    ps_selection = ps.register_point_cloud("selection", np.zeros((1, 3)), radius=pointradius, color=(0, 1, 0), enabled=False)
    ps_selection.set_radius(pointradius, relative=False)

    ps_selection.set_ignore_slice_plane(ps_plane, True)


    ps.set_invoke_user_callback_for_nested_show(True)
    ps.set_user_callback(callback)
    ps.show()