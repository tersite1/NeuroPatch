import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import (
    ViTModel,
    ViTEncoder,
    ViTLayer,
    ViTSelfAttention,
    ViTIntermediate,
    ViTOutput
)

class CustomViTSelfAttention(ViTSelfAttention):
    def forward(self, hidden_states, head_mask=None, output_attentions=False, binary_mask=None):
        B, N, C = hidden_states.shape
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self.transpose_for_scores(query)  # (B, num_heads, N, head_dim)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (B, num_heads, N, N)
        attention_scores = attention_scores / self.attention_head_size**0.5

        # === Sparse Attention 적용 ===
        if binary_mask is not None:
            mask = binary_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class CustomViTLayer(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CustomViTSelfAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False, binary_mask=None):
        self_attention_outputs = self.attention(
            hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            binary_mask=binary_mask
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # attention weights, if needed

        hidden_states = self.layernorm_before(hidden_states + attention_output)
        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,) + outputs

class CustomViTEncoder(ViTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CustomViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, head_mask=None, output_attentions=False, binary_mask=None):
        all_attentions = [] if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                head_mask=head_mask[i] if head_mask is not None else None,
                output_attentions=output_attentions,
                binary_mask=binary_mask
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions.append(layer_outputs[1])
        return (hidden_states, all_attentions) if output_attentions else (hidden_states,)

class CustomViTModel(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CustomViTEncoder(config)

    def forward(self, pixel_values=None, patch_embeddings=None, binary_mask=None, **kwargs):
        if patch_embeddings is not None:
            embedding_output = patch_embeddings
        else:
            embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=kwargs.get("head_mask", None),
            output_attentions=kwargs.get("output_attentions", False),
            binary_mask=binary_mask,
        )
        return encoder_outputs[0]
