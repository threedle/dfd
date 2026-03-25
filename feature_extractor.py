"""
Generic FeatureExtractor wrapper class hierarchy.
Replaces the args.model if/elif chain in distillation_v3.py.
Each model gets its own subclass that wraps the corresponding get_pixel_features_* function.
"""
import torch
import gc


class FeatureExtractor:
    """Base class for all feature extraction models."""

    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._kwargs = kwargs

    def init_model(self):
        """Initialize the underlying model. Called once."""
        raise NotImplementedError

    def get_pixel_features(self, renderings, H, W, **kwargs):
        """
        Extract per-pixel features from rendered views.

        Args:
            renderings: (B, H, W, 4) rendered images
            H, W: image dimensions
            **kwargs: model-specific args (batch_size, normalize, debug, etc.)

        Returns:
            view_features: list of feature tensors or a single tensor
        """
        raise NotImplementedError

    def cleanup(self):
        """Free GPU memory after feature extraction."""
        if self.model is not None:
            try:
                self.model = self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def needs_normal_map(self):
        """Whether this model requires normal map renderings."""
        return False

    @property
    def supports_sam_refinement(self):
        """Whether SAM-based feature refinement is applicable to this model."""
        return False

class DINOv2Extractor(FeatureExtractor):
    """DINOv2 feature extraction."""

    def init_model(self, arch=None):
        if self.model is not None:
            return
        from dino import init_dino
        arch = self._kwargs["arch"]
        if arch is None:
            print("Warning: arch is not provided for dino2. Using default arch: dinov2_vitg14_reg")
            arch = "dinov2_vitg14_reg"
        self.model = init_dino(self.device, archtype=arch)
        self.model_name = arch

    @property
    def supports_sam_refinement(self):
        return True

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=10, normalize=True,
                           debug=False, **kwargs):
        from pixel_features import get_pixel_features_dino
        return get_pixel_features_dino(
            self.device, self.model, renderings,
            H=H, W=W,
            normalize=normalize,
            debug=debug,
            batch_size=batch_size,
        )


class DINOv3Extractor(FeatureExtractor):
    """ DINOv3 feature extraction.
    Required kwargs: repodir, checkpoint
    """

    def init_model(self, arch=None, checkpoint=None, repodir=None):
        if self.model is not None:
            return
        from dino import init_dino

        arch = self._kwargs["arch"]
        if arch is None:
            print("Warning: arch is not provided for dino3. Using default arch: dinov3_vit7b16")
            arch = "dinov3_vit7b16"
        checkpoint = self._kwargs["checkpoint"]
        if checkpoint is None:
            print("Warning: checkpoint is not provided for dino3. Using default checkpoint: /net/projects2/ranalab/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
            checkpoint = "/net/projects2/ranalab/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
        repodir = self._kwargs["repodir"]
        if repodir is None:
            print("Warning: repodir is not provided for dino3. Using default repodir: /net/projects/ranalab/guanzhi/dinov3")
            repodir = "/net/projects/ranalab/guanzhi/dinov3"
        self.model = init_dino(
            self.device,
            repodir=repodir,
            archtype=arch,
            source="local",
            weights=checkpoint,
        )
        self.model_name = arch

    @property
    def supports_sam_refinement(self):
        return True

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=10, normalize=True, half=True,
                           debug=False, **kwargs):
        from pixel_features import get_pixel_features_dino3
        return get_pixel_features_dino3(
            self.device, self.model, renderings,
            H=H, W=W,
            normalize=normalize,
            half=half,
            debug=debug,
            batch_size=batch_size,
        )


class RADIOExtractor(FeatureExtractor):
    """RADIO feature extraction."""

    def init_model(self, arch=None):
        if self.model is not None and getattr(self, "image_processor", None) is not None:
            return
        from transformers import AutoModel, CLIPImageProcessor

        arch = self._kwargs["arch"]
        if arch is None:
            print("Warning: arch is not provided for radio. Using default arch: C-RADIOv3-g")
            arch = "C-RADIOv3-g"

        self.image_processor = CLIPImageProcessor.from_pretrained(f"nvidia/{arch}")
        self.model = AutoModel.from_pretrained(f"nvidia/{arch}", trust_remote_code=True)
        self.model.eval().to(self.device)
        self.model_name = f"radio_{arch}"

    @property
    def supports_sam_refinement(self):
        return True

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=10, normalize=True, half=True,
                           debug=False, **kwargs):
        from pixel_features import get_pixel_features_radio
        return get_pixel_features_radio(
            self.device,
            self.model,
            self.image_processor,
            renderings,
            H=H, W=W,
            normalize=normalize,
            half=half,
            debug=debug,
            batch_size=batch_size,
        )

    def cleanup(self):
        # `image_processor` is CPU-only but can hold large configs/caches; clear it too.
        if hasattr(self, "image_processor"):
            del self.image_processor
            self.image_processor = None
        super().cleanup()


class SAMExtractor(FeatureExtractor):
    """SAM v1 feature extraction."""

    def init_model(self, arch=None, checkpoint=None):
        if self.model is not None:
            return
        from segment_anything import SamPredictor, sam_model_registry

        arch = self._kwargs["arch"]
        if arch is None:
            print("Warning: arch is not provided for sam. Using default arch: vit_l")
            arch = "vit_l"
        checkpoint = self._kwargs["checkpoint"]
        if checkpoint is None:
            print("Warning: checkpoint is not provided for sam. Using default checkpoint: /net/scratch/rliu/SAMmodels/sam_vit_l_0b3195.pth")
            checkpoint = "/net/scratch/rliu/SAMmodels/sam_vit_l_0b3195.pth"

        sam = sam_model_registry[arch](checkpoint=checkpoint)
        self.model = SamPredictor(sam)
        self.model.model = self.model.model.to(self.device)
        self.model_name = f"sam_{arch}"

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=5, normalize=True, half=True,
                           debug=False, **kwargs):
        from pixel_features import get_pixel_features_sam
        return get_pixel_features_sam(
            self.device, self.model, renderings,
            normalize=normalize,
            half=half,
            debug=debug,
            batch_size=batch_size,
        )

    def cleanup(self):
        super().cleanup()


class SAM2Extractor(FeatureExtractor):
    """SAM2 feature extraction."""

    def init_model(self, repodir=None, checkpoint=None, model_cfg=None):
        import sys
        import os

        if self.model is not None:
            return

        repodir = self._kwargs["repodir"]
        if repodir is None:
            print("Warning: repodir is not provided for sam2. Using default repodir: /net/projects/ranalab/guanzhi/DFD/sam2_repo")
            repodir = "/net/projects/ranalab/guanzhi/DFD/sam2_repo"

        if not os.path.isdir(repodir):
            raise FileNotFoundError(
                f"SAM2 repo not found at {repodir}. "
                f"Set repodir=... when creating the extractor, or create/clone sam2_repo."
            )
        if repodir not in sys.path:
            sys.path.append(repodir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = self._kwargs["checkpoint"]
        if checkpoint is None:
            print("Warning: checkpoint is not provided for sam2. Using default checkpoint: /net/projects/ranalab/guanzhi/DFD/sam2_repo/checkpoints/sam2.1_hiera_large.pt")
            checkpoint = os.path.join(repodir, "checkpoints/sam2.1_hiera_large.pt")

        model_cfg = self._kwargs["model_cfg"]
        if model_cfg is None:
            print("Warning: model_cfg is not provided for sam2. Using default local model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=20, normalize=True, half=True,
                           debug=False, concat_hr=False, **kwargs):
        from pixel_features import get_pixel_features_sam2
        return get_pixel_features_sam2(
            self.device, self.model, renderings,
            normalize=normalize,
            half=half,
            debug=debug,
            batch_size=batch_size,
            concat_hr=concat_hr,
        )

    def cleanup(self):
        super().cleanup()


class CLIPExtractor(FeatureExtractor):
    """CLIP feature extraction."""

    def init_model(self, arch=None, checkpoint=None):
        if self.model is not None:
            return
        from clip_stuff import CLIPConvFeatures

        arch = self._kwargs["arch"]
        if arch is None:
            print("Warning: arch is not provided for clip. Using default arch: ViT-L-14")
            arch = "ViT-L-14"

        checkpoint = self._kwargs["checkpoint"]
        if checkpoint is None:
            print("Warning: checkpoint is not provided for clip. Using default checkpoint: /net/scratch/rliu/CLIPmodels/ViT-L-14.pt")
            checkpoint = "/net/scratch/rliu/CLIPmodels/ViT-L-14.pt"

        self.model = CLIPConvFeatures(
            device=self.device,
            clip_model_name=arch,
            clip_model_path=checkpoint,
            num_augs=0,
        )
        self.model_name = f"clip_{arch}"

    def get_pixel_features(self, renderings, H, W, *,
                           batch_size=20, normalize=True, half=False,
                           debug=False, **kwargs):
        from pixel_features import get_pixel_features_clip
        return get_pixel_features_clip(
            self.device, self.model, renderings,
            normalize=normalize,
            half=half,
            debug=debug,
            batch_size=batch_size,
        )

    def cleanup(self):
        super().cleanup()


# --- Registry and factory ---

EXTRACTORS = {
    "dino2": DINOv2Extractor,
    "dino3": DINOv3Extractor,
    "radio": RADIOExtractor,
    "sam": SAMExtractor,
    "sam2": SAM2Extractor,
    "clip": CLIPExtractor,
}


def create_extractor(model_name, device, arch=None, checkpoint=None, repodir=None, model_cfg=None):
    """
    Factory function to create and initialize a FeatureExtractor.

    Args:
        model_name: one of 'dino2', 'dino3', 'radio', 'sam', 'sam2', 'clip'
        device: torch device
        arch: architecture name
        checkpoint: checkpoint path
        repodir: repository directory
        model_cfg: model configuration file

    Returns:
        FeatureExtractor instance with model loaded
    """
    if model_name not in EXTRACTORS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {sorted(set(list(EXTRACTORS.keys())))}"
        )

    extractor = EXTRACTORS[model_name](model_name, device, arch=arch, checkpoint=checkpoint, repodir=repodir, model_cfg=model_cfg)
    extractor.init_model()
    return extractor

