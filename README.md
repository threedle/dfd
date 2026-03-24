# Deep Feature Deformation Weights

<a href="https://arxiv.org/abs/2601.12527"><img src="https://img.shields.io/badge/arXiv-2601.12527-b31b1b.svg" height="22"></a>
<a href="https://threedle.github.io/dfd"><img src="https://img.shields.io/badge/Project-Page-blue.svg" height="22"></a>

<img src="assets/teaser.png" width="100%">

Official implementation of CVPR 2026 paper "Deep Feature Deformation Weights".

## Prerequisites

<!-- TODO: key version constraints -->

### Installation

<!-- TODO: environment.yml -->

```bash
conda env create -f environment.yml
conda activate dfd
```

## Barycentric Feature Distillation

You can call `barycentric_distillation()` in `distillation.py` to perform feature distillation, which distills per-pixel image features into a feature field via barycentric sampling. The script can be called directly using the example asset as follows:

```bash
python distillation.py assets/crab/crab.glb outputs/crab --model dino2 --arch dinov2_vitg14_reg
```

Outputs are saved to `outputs/crab/`, including the trained MLP (`final_encoder.pth`), features sampled on mesh vertices (`crab.pt`), and PCA-colored screenshots.

### Command Line Arguments
Objs with texture coordinates can be rendered with texture using the `--texturedir` kwarg.

High resolution meshes (>100k vertices) should be first downsampled before rendering by using the `--reduction` kwarg. This calls either GLTF-transform (for GLBs) or QEM decimation (other formats) under the hood. The distillation is guaranteed to be fast (<5 min) for any choice of encoder, but for high resolution shapes the rendering becomes a non-trivial bottleneck.

`--use_sam` will refine the render feature maps by using SAM segmentation to identify areas where patch-level features "bleed" into other parts of the shape and fix them. This will improve distilled feature quality but significantly improve runtime.

| Argument | Default | Description |
|---|---|---|
| `--texturedir` | `None` | Path to texture image for textured OBJ rendering. GLB models will automatically have their textures extracted using trimesh. |
| `--saveto` | `vertices` | Saves output feature tensor sampled either over vertices or face centroids. |
| `--no_cache` | `False` | Skip writing cached intermediates to disk (saves disk space) |
| `--reduction` | `0` | Fraction of edges to collapse for mesh simplification before processing |

**Image model parameters**

| Argument | Default | Description |
|---|---|---|
| `--model` | `dino2` | Feature extractor: `dino2`, `dino3`, `diff3f`, `clip`, `sam`, `sam2`, `radio` |
| `--arch` | `None` | Architecture name for the model (e.g. `dinov2_vitg14_reg`) |
| `--checkpoint` | `None` | Path to local model checkpoint weights |
| `--repodir` | `None` | Path to local repository source for the model |
| `--model_cfg` | `None` | Path to model config file (only relevant for `sam2` model) |
| `--sam2_hr` | `False` | Concatenate high-resolution SAM2 features |

**Training parameters**

| Argument | Default | Description |
|---|---|---|
| `-H` / `--imgh` | `512` | Render image height |
| `-W` / `--imgw` | `512` | Render image width |
| `--nviews` | `24` | Total views to render (should be a multiple of 3 for default viewtype) |
| `--viewtype` | `default` | View sampling strategy: `default` or `fib` |
| `--batchsize` | `2` | Views to batch during feature field optimization |
| `--viewbatchsize` | `16` | Number of views to batch for rendering |
| `--featurebatchsize` | `2` | Number of views to batch for feature extraction |
| `--lr` | `1e-3` | Learning rate |
| `--iters` | `25` | Training iterations |

**Gaussian blurring**

| Argument | Default | Description |
|---|---|---|
| `--noiseradius` | `0.05` | Maximum radius for sampling noisy positions about each training position |
| `--noisen` | `0` | Number of noise samples per position sample (disabled by default) |

**SAM feature reassignment**

| Argument | Default | Description |
|---|---|---|
| `--use_sam` | `0` | If > 0, uses SAM masks to refine per-pixel features and reduce feature bleeding. Value determines the outlier threshold (L2 distance). |

**MLP parameters**

| Argument | Default | Description |
|---|---|---|
| `--nlayers` | `4` | Number of MLP layers |
| `--width` | `256` | Hidden layer width |
| `--positional_encoding` | `False` | Use Fourier features for positional encoding |
| `--sigma` | `5.0` | Sigma for Fourier features |

## Interactive Deformation GUI

`interactive_affine.py` launches a [Polyscope](https://polyscope.run/)-based GUI for real-time handle-based mesh deformation using the distilled feature weights.

**Click any vertex** to place a handle and visualize its influence weights over the surface. Then use the translation, rotation, and scale sliders to deform the mesh.

### Capabilities

<table>
<tr>
<td align="center"><video src="https://github.com/user-attachments/assets/1aa412b5-566d-42e5-8b77-c9eed976e91a" width="100%"></video>Mesh with topological defects.</td>
<td align="center"><video src="https://github.com/user-attachments/assets/ff1c21c2-1b88-4c46-b633-26083ca4e4e6" width="100%"></video>Symmetry, feature anchors, locality weighting.</td>
</tr>
</table>

### Example

Run the interactive GUI using the weights distilled above:

```bash
python interactive_affine.py assets/crab/crab.glb --weightsdir outputs/crab/final_encoder.pth
```

## Citation

```bibtex
@InProceedings{DFD_Liu_2026_CVPR,
      author = {Liu, Richard and Lang, Itai and Hanocka, Rana},
      title = {Deep Feature Deformation Weights},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2026}
    }
```
