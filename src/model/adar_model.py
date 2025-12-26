"""
Final ADAR Model Implementation
Aspect Detection with Attention Reorganization

This is the complete ADAR model that integrates:
1. BART multimodal encoder
2. Dimension alignment layer
3. OT-based alignment (Sinkhorn + optimal transport)
4. BART decoder

Key Innovation:
- OT-based alignment replaces AoM's noun_attention + multimodal_GCN
- Dimension alignment: [B, 66, 768] -> H_tilde: [B, 2m, 768]
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.config import MultiModalBartConfig
from src.model.modeling_bart import PretrainedBartModel, BartModel
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span
from src.model.ot_alignment import CoarseGrainedAlignment
from transformers import BartTokenizer


class DimensionAlignmentLayer(nn.Module):
    """
    Dimension Alignment Layer: Convert AoM's [B, 66, 768] to ADAR's [B, 2m, 768]
    """
    def __init__(self, input_dim: int, target_dim: int, output_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.output_dim = output_dim

        self.linear_projection = nn.Linear(input_dim, target_dim)
        self.feature_fusion = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 66, 768] (AoM format)
        Output: [B, 2m, 768] (ADAR format - H_tilde)
        """
        # Linear projection to target dimension
        x_projected = self.linear_projection(x.transpose(-2, -1)).transpose(-2, -1)

        # Feature fusion
        x_fused = self.feature_fusion(x_projected)

        # Layer normalization
        x_normalized = self.layer_norm(x_fused)

        return x_normalized


class ADARModelFinal(PretrainedBartModel):
    """
    Final ADAR Model

    Complete implementation with:
    - BART multimodal encoder/decoder
    - Dimension alignment layer
    - OT-based alignment (replaces noun_attention + GCN)
    """

    def __init__(
        self,
        config: MultiModalBartConfig,
        args,
        bart_model: str,
        tokenizer: BartTokenizer,
        label_ids,
        target_dim: int = 40,  # 2m
        use_ot_alignment: bool = True,
        ot_epsilon: float = 0.1,
        ot_max_iter: int = 100
    ):
        super().__init__(config)
        self.config = config
        self.args = args
        self.mydevice = args.device
        self.use_ot_alignment = use_ot_alignment

        # ==================== BART Architecture ====================
        bart_model_instance = BartModel.from_pretrained(bart_model)
        num_tokens, _ = bart_model_instance.encoder.embed_tokens.weight.shape
        bart_model_instance.resize_token_embeddings(
            len(tokenizer.unique_no_split_tokens) + num_tokens
        )

        encoder = bart_model_instance.encoder
        decoder = bart_model_instance.decoder

        padding_idx = config.pad_token_id
        encoder.embed_tokens.padding_idx = padding_idx

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(
                    tokenizer._base_tokenizer.tokenize(token)
                )
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(
                    _tokenizer.tokenize(token[2:-2])
                )
                embed = bart_model_instance.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += bart_model_instance.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                bart_model_instance.decoder.embed_tokens.weight.data[index] = embed

        # ==================== Multimodal Encoder ====================
        self.multimodal_encoder = MultiModalBartEncoder(
            config,
            encoder,
            tokenizer.img_feat_id,
            tokenizer.cls_token_id
        )

        # ==================== Dimension Alignment Layer ====================
        self.input_dim = 66  # AoM: 51 image + 15 text
        self.target_dim = target_dim  # ADAR: 2m
        self.dimension_alignment = DimensionAlignmentLayer(
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            output_dim=config.d_model
        )

        # ==================== OT-based Alignment ====================
        if self.use_ot_alignment:
            self.ot_alignment = CoarseGrainedAlignment(
                embed_dim=config.d_model,
                num_heads=4,
                epsilon=ot_epsilon,
                max_iter=ot_max_iter
            )

        # ==================== Decoder ====================
        causal_mask = torch.zeros(512, 512).fill_(float('-inf')).triu(diagonal=1)
        self.causal_mask = causal_mask

        self.decoder = MultiModalBartDecoder_span(
            self.config,
            tokenizer,
            decoder,
            tokenizer.pad_token_id,
            label_ids,
            self.causal_mask,
            args.gcn_on,
            need_tag=True,
            only_sc=False
        )

        # Loss function
        from src.model.modules import Span_loss
        self.span_loss_fct = Span_loss()

    def prepare_state(
        self,
        input_ids,
        image_features,
        noun_mask,
        attention_mask=None,
        dependency_matrix=None,
        sentiment_value=None,
        first=None
    ):
        """
        Prepare state with OT-based alignment
        """
        # Multimodal encoding
        encoder_output = self.multimodal_encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        encoder_outputs = encoder_output.last_hidden_state
        hidden_states = encoder_output.hidden_states
        encoder_mask = attention_mask

        # Dimension alignment
        H_tilde = self.dimension_alignment(encoder_outputs)

        # OT-based alignment (replaces noun_attention + GCN)
        if self.use_ot_alignment:
            # Split visual and textual features
            visual_features = encoder_outputs[:, :51, :]  # (B, 51, 768)
            textual_features = encoder_outputs[:, 51:, :]  # (B, 15, 768)

            # Apply OT alignment
            aligned_features = self.ot_alignment(visual_features, textual_features)
            H_tilde = aligned_features

        # Build state
        state = self._build_bart_state(
            encoder_output=H_tilde,
            encoder_mask=encoder_mask,
            src_tokens=input_ids[:, 51:],
            first=first,
            src_embed_outputs=hidden_states[0],
            mix_feature=None  # Not using GCN anymore
        )

        return state

    def _build_bart_state(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs, mix_feature):
        """Build BartState-compatible state object"""
        from src.model.model import BartState
        return BartState(
            encoder_output=encoder_output,
            encoder_mask=encoder_mask,
            src_tokens=src_tokens,
            first=first,
            src_embed_outputs=src_embed_outputs,
            mix_feature=mix_feature
        )

    def forward(
        self,
        input_ids,
        image_features,
        sentiment_value,
        noun_mask,
        attention_mask=None,
        dependency_matrix=None,
        aesc_infos=None,
        encoder_outputs: Optional[Tuple] = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        """
        Forward pass
        """
        # Prepare state
        state = self.prepare_state(
            input_ids, image_features, noun_mask,
            attention_mask, dependency_matrix, sentiment_value
        )

        # Decode
        spans, span_mask = [
            aesc_infos['labels'].to(input_ids.device),
            aesc_infos['masks'].to(input_ids.device)
        ]
        logits = self.decoder(spans, state, sentiment_value)

        # Compute loss
        loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

        return loss


class ADARModelConfig:
    """
    Configuration class for ADAR model
    """
    def __init__(
        self,
        target_dim: int = 40,
        use_ot_alignment: bool = True,
        ot_epsilon: float = 0.1,
        ot_max_iter: int = 100,
        embed_dim: int = 768
    ):
        self.target_dim = target_dim
        self.use_ot_alignment = use_ot_alignment
        self.ot_epsilon = ot_epsilon
        self.ot_max_iter = ot_max_iter
        self.embed_dim = embed_dim

    def __repr__(self):
        return (f"ADARModelConfig("
                f"target_dim={self.target_dim}, "
                f"use_ot_alignment={self.use_ot_alignment}, "
                f"ot_epsilon={self.ot_epsilon}, "
                f"ot_max_iter={self.ot_max_iter})")


def create_adar_model(
    config: MultiModalBartConfig,
    args,
    bart_model: str,
    tokenizer: BartTokenizer,
    label_ids,
    target_dim: int = 40,
    use_ot_alignment: bool = True,
    ot_epsilon: float = 0.1,
    ot_max_iter: int = 100
) -> ADARModelFinal:
    """
    Factory function to create ADAR model

    Args:
        config: BART configuration
        args: Training arguments
        bart_model: BART pretrained model name
        tokenizer: Tokenizer instance
        label_ids: List of label IDs
        target_dim: Target dimension (2m)
        use_ot_alignment: Whether to use OT alignment
        ot_epsilon: Sinkhorn regularization parameter
        ot_max_iter: Maximum Sinkhorn iterations

    Returns:
        ADARModelFinal instance
    """
    return ADARModelFinal(
        config=config,
        args=args,
        bart_model=bart_model,
        tokenizer=tokenizer,
        label_ids=label_ids,
        target_dim=target_dim,
        use_ot_alignment=use_ot_alignment,
        ot_epsilon=ot_epsilon,
        ot_max_iter=ot_max_iter
    )


def compare_models():
    """
    Comparison between AoM and ADAR models
    """
    comparison = {
        "Component": [
            "Encoder",
            "Decoder",
            "Dimension Alignment",
            "Visual-Textual Alignment",
            "Graph Neural Networks",
            "Attention Mechanism"
        ],
        "AoM": [
            "BART Multimodal Encoder",
            "BART Decoder",
            "None (direct 66-dim)",
            "noun_attention + multimodal_GCN",
            "GCN + GAT",
            "Multi-head attention"
        ],
        "ADAR": [
            "BART Multimodal Encoder",
            "BART Decoder",
            "DimensionAlignmentLayer (66 -> 2m)",
            "OT-based alignment (Sinkhorn + OT)",
            "None (replaced by OT)",
            "OT alignment + gating"
        ]
    }

    print("=" * 80)
    print("AoM vs ADAR Model Comparison")
    print("=" * 80)

    col_width = [25, 40, 40]
    print(f"{'Component':<{col_width[0]}} {'AoM':<{col_width[1]}} {'ADAR':<{col_width[2]}}")
    print("-" * sum(col_width))

    for i in range(len(comparison["Component"])):
        print(f"{comparison['Component'][i]:<{col_width[0]}} "
              f"{comparison['AoM'][i]:<{col_width[1]}} "
              f"{comparison['ADAR'][i]:<{col_width[2]}}")

    print("=" * 80)
    print("\nKey Differences:")
    print("1. ADAR adds dimension alignment layer (66 -> 2m)")
    print("2. ADAR replaces noun_attention + GCN with OT-based alignment")
    print("3. ADAR uses Sinkhorn algorithm for optimal transport")
    print("4. ADAR removes graph neural networks (simplified architecture)")
    print("=" * 80)


if __name__ == "__main__":
    compare_models()
