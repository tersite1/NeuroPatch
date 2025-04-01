import torch
import torch.nn as nn
from torchvision.models import vgg16
from spikingjelly.activation_based import neuron
from module.vit_custom import CustomViTModel
from transformers import ViTConfig
from backbone.attention_utils import lif_to_attention, apply_attention_mask


class PatchVGGWithLIF(nn.Module):
    def __init__(self, patch_size=16, img_size=224, in_channels=3, embed_dim=768, lif_threshold=1.0,
                 attn_mode='soft', attn_temperature=0.7, masking_mode='scale'):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.attn_mode = attn_mode
        self.attn_temperature = attn_temperature
        self.masking_mode = masking_mode

        # VGG16 초기 블록 (MaxPool 제거)
        full_vgg = vgg16(weights='DEFAULT').features
        self.vgg = nn.Sequential(*[layer for layer in full_vgg if not isinstance(layer, nn.MaxPool2d)])[:10]

        # LIF 뉴런
        self.lif = neuron.LIFNode(v_threshold=lif_threshold, detach_reset=True)

        # LIF 출력 → ViT 입력 차원 정렬
        self.output_proj = nn.Linear(256, embed_dim)

        # Custom ViT 설정
        vit_config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=embed_dim * 4,
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels,
        )
        self.vit = CustomViTModel(vit_config)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_H = H // self.patch_size
        patch_W = W // self.patch_size
        N = patch_H * patch_W

        # unfold로 패치 병렬 추출: (B, C, patch_H, patch_W, pH, pW)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H_p, W_p, C, pH, pW)
        x = x.view(-1, C, self.patch_size, self.patch_size)  # (B*N, C, pH, pW)

        # VGG + LIF + Projection 병렬 처리
        vgg_feat = self.vgg(x)                                 # (B*N, C', h, w)
        avg_feat = vgg_feat.mean(dim=[2, 3])                   # (B*N, C')
        lif_out = self.lif(avg_feat)                           # (B*N, C')
        proj_out = self.output_proj(lif_out)                   # (B*N, D)

        patch_embeddings = proj_out.view(B, N, self.embed_dim)  # (B, N, D)
        lif_outputs = lif_out.view(B, N, -1)                    # (B, N, C')

        # Attention 계산 및 마스킹
        attn_scores = lif_to_attention(lif_outputs, mode=self.attn_mode, temperature=self.attn_temperature)  # (B, N)
        masked_embeddings = apply_attention_mask(patch_embeddings, attn_scores, mode=self.masking_mode)      # (B, N, D)

        # ViT forward
        vit_out = self.vit(patch_embeddings=masked_embeddings).last_hidden_state  # (B, N, D)
        return vit_out
