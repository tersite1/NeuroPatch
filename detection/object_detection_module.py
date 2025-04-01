import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=10, num_queries=10, depth=2, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))  # (Q, D)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # GRU-based query refinement
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # Attention pooling (query-aware global memory pooling)
        self.attn_proj_q = nn.Linear(embed_dim, embed_dim)
        self.attn_proj_k = nn.Linear(embed_dim, embed_dim)
        self.attn_proj_v = nn.Linear(embed_dim, embed_dim)

        # Conv-BN-ReLU blocks with residual connection
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim * 2, kernel_size=1),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim * 2, kernel_size=1),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU()
        )

        # Final classifier and regressor (takes fused features)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.bbox_regressor = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
  

    def forward(self, x, labels):
        # x: (B, N, D) ← patch embeddings
        B = x.size(0)
        memory = x  # (B, N, D)

        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        query = self.decoder(query, memory)  # (B, Q, D)

        # GRU refinement
        query, _ = self.gru(query)  # (B, Q, D)

        # Attention pooling: global memory summary per query
        q = self.attn_proj_q(query)       # (B, Q, D)
        k = self.attn_proj_k(memory)      # (B, N, D)
        v = self.attn_proj_v(memory)      # (B, N, D)

        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (B, Q, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, Q, N)
        global_context = torch.matmul(attn_weights, v)  # (B, Q, D)

        # Fusion: [query || global_context] → (B, Q, 2D)
        fused = torch.cat([query, global_context], dim=-1)  # (B, Q, 2D)

        # Conv-BN-ReLU blocks with residual (reshape to (B, 2D, Q) for Conv1d)
        x_conv = fused.transpose(1, 2)  # (B, 2D, Q)
        residual = x_conv
        x_conv = self.conv1(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = x_conv + residual  # Residual connection
        fused_conv = x_conv.transpose(1, 2)  # (B, Q, 2D)

        cls_logits = self.classifier(fused_conv)  # (B, Q, C)
        bboxes = self.bbox_regressor(fused_conv)  # (B, Q, 4)

        # Reshape to match cross_entropy
        B, Q, C = cls_logits.size()
        cls_logits = cls_logits.view(-1, C)  # (B * Q, C)
        bboxes = self.bbox_regressor(fused_conv)  # (B * Q)

        # Compute loss
        cls_loss = self.cls_loss(cls_logits, labels)  # (B * Q, C), (B * Q)
        return cls_logits, bboxes