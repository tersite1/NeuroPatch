import torch.nn as nn
from torchvision.models import resnet18, efficientnet_b0, convnext_tiny
from backbone.vit_patch_vgg_lif import PatchVGGWithLIF


def get_backbone_model(model_type, img_size=224, patch_size=16, embed_dim=768, pretrained=False):
    if model_type.lower() == 'vgg':
        return PatchVGGWithLIF(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

    elif model_type.lower() == 'resnet18':
        resnet = resnet18(pretrained=pretrained)
        layers = list(resnet.children())[:6]  # conv1 ~ layer2
        feature_extractor = nn.Sequential(*layers)
        return nn.Sequential(
            feature_extractor,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    elif model_type.lower() == 'efficientnet_b0':
        effnet = efficientnet_b0(pretrained=pretrained)
        return nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(effnet.classifier[1].in_features, embed_dim)
        )

    elif model_type.lower() == 'convnext_tiny':
        convnext = convnext_tiny(pretrained=pretrained)
        return nn.Sequential(
            convnext.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(convnext.classifier[2].in_features, embed_dim)
        )

    else:
        raise ValueError(f"Unsupported backbone type: {model_type}")