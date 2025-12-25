"""
ADAR Model Architecture - 新版本
基于AoM框架实现新的数据流

新数据流:
输入 → ResNet → [B, 2048, 7, 7] → [B, 49, 2048] → ImageEmbedding → [B, 51, 768]
    → 文本嵌入 → [B, 15, 768] → 多模态编码 → [B, 66, 768]
    → 分离: 视觉[B, 51, 768] + 文本[B, 15, 768]
    → 粗粒度OT对齐 → [B, 15, 768]  (对齐视觉到文本空间)
    → AFB过滤 → [B, 51, 768]      (过滤视觉噪声，保留维度)
    → 特征拼接 → [B, 66, 768]    (垂直拼接: 51+15)
    → ADRM精炼 → [B, 66, 768]    (2层跨模态注意力)
    → 特征融合 → [B, 66, 768]    (加权融合)
    → 解码器 → 输出

关键组件:
1. AFBModule: 自适应过滤仓，过滤视觉噪声
2. ADRMRefinement: 2层跨模态注意力精炼模块
3. FeatureFusion: 特征融合层
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.config import MultiModalBartConfig
from src.model.modeling_bart import PretrainedBartModel, BartModel
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span
from src.model.ot_alignment import CoarseGrainedAlignment
from src.model.afb import AFBModule
from src.model.adrm_refinement import ADRMRefinement
from src.model.feature_fusion import FeatureFusion
from transformers import BartTokenizer


class ADARModelNew(PretrainedBartModel):
    """
    ADAR (Aspect Detection with Attention Reorganization) Model - 新版本
    基于新数据流重构的ADAR模型
    """
    def __init__(
        self,
        config: MultiModalBartConfig,
        args,
        bart_model: str,
        tokenizer: BartTokenizer,
        label_ids,
        visual_len: int = 51,
        textual_len: int = 15
    ):
        super().__init__(config)
        self.config = config
        self.args = args
        self.mydevice = args.device
        self.visual_len = visual_len
        self.textual_len = textual_len

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

        # ==================== 新数据流组件 ====================

        # 1. OT-based Alignment Module (保持不变)
        self.ot_alignment = CoarseGrainedAlignment(
            embed_dim=config.d_model,
            num_heads=4,
            epsilon=0.1,
            max_iter=100
        )

        # 2. AFB Module - 自适应过滤仓
        self.afb_module = AFBModule(
            embed_dim=config.d_model,
            dropout=0.1
        )

        # 3. ADRM Refinement Module - 2层跨模态注意力
        self.adrm_refinement = ADRMRefinement(
            embed_dim=config.d_model,
            num_heads=8,
            dropout=0.1,
            visual_len=visual_len,
            textual_len=textual_len
        )

        # 4. Feature Fusion Module - 特征融合层
        self.feature_fusion = FeatureFusion(
            embed_dim=config.d_model,
            visual_len=visual_len,
            textual_len=textual_len,
            dropout=0.1
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

    def expand_aligned_features(self, aligned_features, target_dim):
        """
        将对齐特征扩展到目标维度

        Args:
            aligned_features: 对齐特征 [B, textual_len, embed_dim]
            target_dim: 目标维度 (visual_len)

        Returns:
            expanded_features: 扩展特征 [B, target_dim, embed_dim]
        """
        batch_size, _, embed_dim = aligned_features.shape

        # 方法1: 简单重复
        expanded = aligned_features.repeat(1, target_dim // self.textual_len + 1, 1)
        expanded = expanded[:, :target_dim, :]

        return expanded

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
        新数据流的状态准备:
        1. 多模态编码
        2. 分离视觉和文本特征
        3. 粗粒度OT对齐 (视觉→文本空间)
        4. AFB过滤 (视觉特征)
        5. 特征拼接 (对齐文本+过滤视觉)
        6. ADRM精炼 (2层跨模态注意力)
        7. 特征融合 (对齐与精炼特征)
        """
        # 多模态编码
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

        # ==================== 新数据流 ====================

        # 步骤1: 分离视觉和文本特征
        visual_features = encoder_outputs[:, :self.visual_len, :]  # [B, 51, 768]
        textual_features = encoder_outputs[:, self.visual_len:, :]  # [B, 15, 768]

        # 步骤2: 粗粒度OT对齐 (视觉→文本空间)
        aligned_visual = self.ot_alignment(visual_features, textual_features)  # [B, 15, 768]

        # 步骤3: AFB过滤 (过滤视觉噪声，保留维度)
        filtered_visual = self.afb_module(visual_features)  # [B, 51, 768]

        # 步骤4: 特征拼接
        # 将aligned_visual [15] 扩展到 [51] 然后与 filtered_visual [51] 垂直拼接
        expanded_aligned = self.expand_aligned_features(aligned_visual, target_dim=self.visual_len)
        concatenated = torch.cat([expanded_aligned, textual_features], dim=1)  # [B, 66, 768]

        # 步骤5: ADRM精炼 (2层跨模态注意力)
        refined_features = self.adrm_refinement(concatenated)  # [B, 66, 768]

        # 步骤6: 特征融合 (对齐与精炼特征)
        fused_features = self.feature_fusion(aligned_visual, refined_features)  # [B, 66, 768]

        # 构建状态
        # 计算mix_feature (如果需要)
        if hasattr(self.args, 'gcn_on') and self.args.gcn_on:
            # 简单的mix_feature计算
            mix_feature = torch.mean(fused_features, dim=1, keepdim=True)
        else:
            mix_feature = None

        state = self._build_bart_state(
            encoder_output=fused_features,
            encoder_mask=encoder_mask,
            src_tokens=input_ids[:, self.visual_len:],
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
        # Prepare state (includes new data flow)
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


def create_adar_model_new(
    config,
    args,
    bart_model='facebook/bart-base',
    tokenizer=None,
    label_ids=None,
    visual_len=51,
    textual_len=15
):
    """
    Factory function to create ADARModelNew
    """
    return ADARModelNew(
        config=config,
        args=args,
        bart_model=bart_model,
        tokenizer=tokenizer,
        label_ids=label_ids,
        visual_len=visual_len,
        textual_len=textual_len
    )