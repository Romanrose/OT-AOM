"""
ADAR Model Architecture
Based on AoM framework for multimodal aspect-based sentiment analysis

Key Modifications:
1. BART Encoder Output: [B, 66, 768] -> H_tilde: [B, 2m, 768]
2. Retain ResNet image encoder
3. Retain all data processing modules
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
    m is a target dimension parameter
    """
    def __init__(self, input_dim: int, target_dim: int, output_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim  # 66 in AoM
        self.target_dim = target_dim  # 2m in ADAR
        self.output_dim = output_dim  # 768

        # Use linear transformation and resampling for dimension conversion
        self.linear_projection = nn.Linear(input_dim, target_dim)
        self.feature_fusion = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 66, 768] (AoM format)
        Output: [B, 2m, 768] (ADAR format - H_tilde)
        """
        B, _, D = x.shape

        # Linear projection to target dimension
        x_projected = self.linear_projection(x.transpose(-2, -1)).transpose(-2, -1)

        # Feature fusion
        x_fused = self.feature_fusion(x_projected)

        # Layer normalization
        x_normalized = self.layer_norm(x_fused)

        return x_normalized


class ADARModel(PretrainedBartModel):
    """
    ADAR (Aspect Detection with Attention Reorganization) Model
    Improved version based on AoM architecture
    """
    def __init__(
        self,
        config: MultiModalBartConfig,
        args,
        bart_model: str,
        tokenizer: BartTokenizer,
        label_ids,
        target_dim: int = 40  # 2m - where m defaults to 20
    ):
        super().__init__(config)
        self.config = config
        self.args = args
        self.mydevice = args.device

        # ==================== Retain AoM's BART Architecture ====================
        # Initialize BART model
        bart_model_instance = BartModel.from_pretrained(bart_model)

        # Get original token embeddings
        num_tokens, _ = bart_model_instance.encoder.embed_tokens.weight.shape
        bart_model_instance.resize_token_embeddings(
            len(tokenizer.unique_no_split_tokens) + num_tokens
        )

        encoder = bart_model_instance.encoder
        decoder = bart_model_instance.decoder

        # Handle padding_idx
        padding_idx = config.pad_token_id
        encoder.embed_tokens.padding_idx = padding_idx

        # Embed special tokens
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

        # ==================== Create Multimodal Encoder ====================
        self.multimodal_encoder = MultiModalBartEncoder(
            config,
            encoder,
            tokenizer.img_feat_id,
            tokenizer.cls_token_id
        )

        # ==================== ADAR Specific: Dimension Alignment Layer ====================
        # AoM input dimension is 66 (51 image + 15 text), convert to ADAR's 2m dimension
        self.input_dim = 66  # AoM standard input dimension
        self.target_dim = target_dim  # ADAR target dimension (2m)
        self.dimension_alignment = DimensionAlignmentLayer(
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            output_dim=config.d_model
        )

        # ==================== ADAR Specific: OT-based Alignment Module ====================
        # Replaces AoM's noun_attention + multimodal_GCN
        self.ot_alignment = CoarseGrainedAlignment(
            embed_dim=config.d_model,
            num_heads=4,
            epsilon=0.1,
            max_iter=100
        )

        # ==================== Retain AoM's Decoder ====================
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

        # ==================== Retain AoM's Other Components ====================
        self.noun_linear = nn.Linear(768, 768)
        self.multi_linear = nn.Linear(768, 768)
        self.att_linear = nn.Linear(768 * 2, 1)
        self.attention = self._build_attention(4, 768, 768)
        self.linear = nn.Linear(768 * 2, 1)
        self.linear2 = nn.Linear(768 * 2, 1)
        self.alpha_linear1 = nn.Linear(768, 768)
        self.alpha_linear2 = nn.Linear(768, 768)

        # GCN Related
        if hasattr(args, 'gcn_on') and args.gcn_on:
            from src.model.GCN import GCN
            self.senti_linear = nn.Linear(768, 768)
            self.context_linear = nn.Linear(768, 768)
            self.mix_linear = nn.Linear(768 * 2, 768)
            self.senti_gcn = GCN(768, 768, 768, dropout=args.gcn_dropout)
            self.context_gcn = GCN(768, 768, 768, dropout=args.gcn_dropout)

        # GAT Related
        from src.model.GAT import GAT
        self.gat = GAT(768, 768, 0.2, 0.2, n_heads=1)
        self.gat_linear = nn.Linear(768, 768)

    def _build_attention(self, num_attention_heads: int, input_size: int, hidden_size: int):
        """Build multi-head attention layer"""
        class Attention(nn.Module):
            def __init__(self, num_attention_heads, input_size, hidden_size):
                super().__init__()
                if hidden_size % num_attention_heads != 0:
                    raise ValueError(
                        f"hidden size {hidden_size} not divisible by "
                        f"number of attention heads {num_attention_heads}"
                    )

                self.num_attention_heads = num_attention_heads
                self.attention_head_size = int(hidden_size / num_attention_heads)
                self.all_head_size = hidden_size

                self.key_layer = nn.Linear(input_size, hidden_size)
                self.query_layer = nn.Linear(input_size, hidden_size)
                self.value_layer = nn.Linear(input_size, hidden_size)

            def trans_to_multiple_heads(self, x):
                new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(new_size)
                return x.permute(0, 2, 1, 3)

            def forward(self, q, k, v):
                key = self.key_layer(k)
                query = self.query_layer(q)
                value = self.value_layer(v)

                key_heads = self.trans_to_multiple_heads(key)
                query_heads = self.trans_to_multiple_heads(query)
                value_heads = self.trans_to_multiple_heads(value)

                attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
                attention_scores = attention_scores / torch.sqrt(
                    torch.tensor(self.attention_head_size, dtype=torch.float)
                )

                attention_probs = F.softmax(attention_scores, dim=-1)
                context = torch.matmul(attention_probs, value_heads)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_size = context.size()[:-2] + (self.all_head_size,)
                context = context.view(*new_size)
                return context

        return Attention(num_attention_heads, input_size, hidden_size)

    def get_noun_embed(self, feature, noun_mask):
        """Extract noun embeddings (maintain AoM logic)"""
        noun_mask = noun_mask.cpu()
        noun_num = [x.numpy().tolist().count(1) for x in noun_mask]
        noun_position = [torch.where(torch.tensor(x) == 1)[0].tolist() for x in noun_mask]

        max_noun_num = max(noun_num) if noun_num else 0
        noun_position = [
            pos + [0] * (max_noun_num - len(pos))
            for pos in noun_position
        ]
        noun_position = torch.tensor(noun_position).to(self.mydevice)

        noun_embed = torch.zeros(feature.shape[0], max_noun_num, feature.shape[-1]).to(self.mydevice)
        for i in range(len(feature)):
            if max_noun_num > 0:
                noun_embed[i] = torch.index_select(feature[i], dim=0, index=noun_position[i])
                noun_embed[i, noun_num[i]:] = torch.zeros(
                    max_noun_num - noun_num[i], feature.shape[-1]
                )
        return noun_embed

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
        Prepare state:
        1. Multimodal encoding
        2. Dimension alignment: [B, 66, 768] -> H_tilde: [B, 2m, 768]
        3. OT-based alignment (replaces noun_attention + GCN)
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

        # ==================== Core Modification: Dimension Alignment ====================
        # AoM original output: [B, 66, 768]
        # ADAR output: H_tilde: [B, 2m, 768]
        H_tilde = self.dimension_alignment(encoder_outputs)

        # ==================== ADAR: OT-based Alignment (Replaces AoM's noun_attention + GCN) ====================
        # Split visual and textual features
        # AoM format: first 51 tokens are image features, rest are text
        visual_features = encoder_outputs[:, :51, :]  # (B, 51, 768)
        textual_features = encoder_outputs[:, 51:, :]  # (B, 15, 768)

        # Apply OT-based coarse-grained alignment
        # This replaces AoM's noun_attention + multimodal_GCN
        aligned_features = self.ot_alignment(visual_features, textual_features)
        # aligned_features: (B, 15, 768)

        # Update H_tilde with aligned features
        H_tilde = aligned_features

        # Build state (maintain BartState interface compatibility)
        state = self._build_bart_state(
            encoder_output=H_tilde,
            encoder_mask=encoder_mask,
            src_tokens=input_ids[:, 51:],
            first=first,
            src_embed_outputs=hidden_states[0],
            mix_feature=mix_feature
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

    def noun_attention(self, encoder_outputs, noun_embed, mode='multi-head'):
        """Noun attention mechanism (maintain AoM logic)"""
        if mode == 'cat':
            multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
            noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
            noun_features_rep = self.noun_linear(noun_features_rep)
            multi_features_rep = self.multi_linear(multi_features_rep)
            concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
            att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
            att_features = torch.matmul(att, noun_embed)

            alpha = torch.sigmoid(self.linear(torch.cat([self.alpha_linear1(encoder_outputs), self.alpha_linear2(att_features)], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)

            encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)
            return encoder_outputs

        elif mode == 'multi-head':
            att_features = self.attention(encoder_outputs, noun_embed, noun_embed)
            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(1 - alpha, encoder_outputs) + torch.mul(alpha, att_features)
            return encoder_outputs

        return encoder_outputs

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
        # Prepare state (includes dimension alignment)
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
        from src.model.modules import Span_loss
        span_loss_fct = Span_loss()
        loss = span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])

        return loss
