import clip
import collections
import torch
import torch.nn as nn
from torchvision import models, transforms

# NOTE: We need to raise the dtype of these otherwise go to inf too easily
def l2_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [torch.square((x_conv - y_conv) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [torch.abs((x_conv - y_conv) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        # NOTE: This may not be true always!! Need to check if use intermediate layers
        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps

class CLIPConvFeatures(torch.nn.Module):
    def __init__(self, clip_model_name="RN101",
                 clip_model_path=None,
                 num_augs = 4,
                 device=torch.device("cuda:0")):
        super(CLIPConvFeatures, self).__init__()
        self.clip_model_name = clip_model_name

        clipload = clip_model_path if clip_model_path else clip_model_name
        self.model, self.clip_preprocess = clip.load(
                    clipload, device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            # init_layers = torch.nn.Sequential(*layers)[:8]
            # self.layer1 = layers[8]
            # self.layer2 = layers[9]
            # self.layer3 = layers[10]
            # self.layer4 = layers[11]
            # self.att_pool2d = layers[12]

            self.layer1 = self.visual_model.layer1
            self.layer2 = self.visual_model.layer2
            self.layer3 = self.visual_model.layer3
            self.layer4 = self.visual_model.layer4
            self.att_pool2d = self.visual_model.attnpool

        self.img_size = self.clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            self.clip_preprocess.transforms[0],  # Resize
            self.clip_preprocess.transforms[1],  # CenterCrop
            self.clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = device
        self.num_augs = num_augs

        augmentations = []
        augmentations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augmentations.append(transforms.RandomResizedCrop(
            self.clip_preprocess.transforms[0].size, scale=(0.8, 0.8), ratio=(1.0, 1.0), antialias=True))
        augmentations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augmentations)

        self.counter = 0

    def forward(self, imgs):
        """
        Parameters
        ----------
        imgs: Torch Tensor [B, C, H, W]
        """
        x = imgs.to(self.device)
        x_augs = []

        # NOTE: First transform from clip preprocess calls resize
        if self.num_augs > 0:
            for n in range(self.num_augs):
                augmented_x = self.augment_trans(self.clip_preprocess.transforms[0](x))
                x_augs.append(augmented_x)
            xs = torch.cat(x_augs, dim=0).to(self.device)
        else:
            xs = self.normalize_transform(x)


        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)

        return xs_fc_features, xs_conv_features

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu1(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
