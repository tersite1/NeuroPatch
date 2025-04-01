import torch
import torch.nn as nn
from torchvision.models import vgg16
from spikingjelly.activation_based import neuron
from transformers import ViTModel, ViTConfig

class PatchVGGWithLIF(nn.Module):
    def __init__(self, patch_size=16, img_size=224, in_channels=3, embed_dim=768, lif_threshold=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # VGG16에서 특징 추출용 Conv+ReLU 블록만 선택 (MaxPool 제거)
        full_vgg = vgg16(weights='DEFAULT').features
        self.vgg = nn.Sequential(*[layer for layer in full_vgg if not isinstance(layer, nn.MaxPool2d)])[:10]  # 2 Conv 블록까지

        # LIF 뉴런
        self.lif = neuron.LIFNode(v_threshold=lif_threshold, detach_reset=True)

        # LIF 출력 → ViT 입력 차원 정렬
        self.output_proj = nn.Linear(256, embed_dim)  # VGG 출력 채널 수에 맞춰 조정 (128 기준)

        # ViT 설정
        vit_config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=embed_dim * 4,
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels,
        )
        self.vit = ViTModel(vit_config)

    def forward(self, x):
        B, C, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        outputs = []
        for i in range(H_p):
            for j in range(W_p):
                patch = x[:, :,
                          i * self.patch_size:(i + 1) * self.patch_size,
                          j * self.patch_size:(j + 1) * self.patch_size]  # [B, 3, 16, 16]

                vgg_feat = self.vgg(patch)           # (B, 128, H', W')
                avg_feat = vgg_feat.mean(dim=[2, 3]) # (B, 128)
                lif_out = self.lif(avg_feat)         # (B, 128)
                proj_out = self.output_proj(lif_out) # (B, embed_dim)
                outputs.append(proj_out)

        patch_embeddings = torch.stack(outputs, dim=1)  # (B, num_patches, embed_dim)

        # ViT attention 수행
        vit_out = self.vit(pixel_values=x).last_hidden_state  # (B, N, D)
        return vit_out
