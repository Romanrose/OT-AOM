"""
ADAR Simple Model - 无AFB版本
简化版ADAR模型，移除AFB模块，只保留核心的对齐和精炼功能

数据流程:
输入 → ResNet → [B, 2048, 7, 7] → [B, 49, 2048] → ImageEmbedding → [B, 51, 768]
    → 文本嵌入 → [B, 15, 768] → 多模态编码 → [B, 66, 768]
    → 分离: 视觉[B, 51, 768] + 文本[B, 15, 768]
    → 粗粒度OT对齐 → [B, 15, 768]  (对齐视觉到文本空间)
    → 特征拼接 → [B, 66, 768]    (垂直拼接: 51+15)
    → ADRM精炼 → [B, 66, 768]    (2层跨模态注意力)
    → 特征融合 → [B, 66, 768]    (加权融合)
    → 解码器 → 输出

组件列表:
1. MultiModalBartEncoder: BART多模态编码器
2. CoarseGrainedAlignment: 粗粒度OT对齐模块 (保留)
3. ADRMRefinement: 2层跨模态注意力精炼模块 (保留)
4. FeatureFusion: 特征融合层 (保留)
5. MultiModalBartDecoder_span: BART多模态解码器

移除的组件:
- AFBModule: 自适应过滤仓 (移除)
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from src.model.config import MultiModalBartConfig
from src.model.modeling_bart import PretrainedBartModel, BartModel
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span
from src.model.ot_alignment import CoarseGrainedAlignment
from src.model.adrm_refinement import ADRMRefinement
from src.model.feature_fusion import FeatureFusion, FeatureFusionV2
from transformers import BartTokenizer


class SimpleADARModelConfig:
    """
    简化版ADAR模型配置类
    移除AFB相关参数，专注于对齐和精炼
    """
    def __init__(
        self,
        # 基础配置
        embed_dim: int = 768,
        visual_len: int = 51,
        textual_len: int = 15,

        # OT对齐配置
        ot_num_heads: int = 4,
        ot_epsilon: float = 0.01,  # 使用优化后的参数
        ot_max_iter: int = 500,    # 使用优化后的参数

        # ADRM精炼配置
        adrm_num_heads: int = 8,
        adrm_dropout: float = 0.1,

        # 特征融合配置
        fusion_method: str = 'weighted_sum',
        fusion_dropout: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.visual_len = visual_len
        self.textual_len = textual_len

        self.ot_num_heads = ot_num_heads
        self.ot_epsilon = ot_epsilon
        self.ot_max_iter = ot_max_iter

        self.adrm_num_heads = adrm_num_heads
        self.adrm_dropout = adrm_dropout

        self.fusion_method = fusion_method
        self.fusion_dropout = fusion_dropout

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'embed_dim': self.embed_dim,
            'visual_len': self.visual_len,
            'textual_len': self.textual_len,
            'ot_num_heads': self.ot_num_heads,
            'ot_epsilon': self.ot_epsilon,
            'ot_max_iter': self.ot_max_iter,
            'adrm_num_heads': self.adrm_num_heads,
            'adrm_dropout': self.adrm_dropout,
            'fusion_method': self.fusion_method,
            'fusion_dropout': self.fusion_dropout
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimpleADARModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)


class SimpleADARModel(PretrainedBartModel):
    """
    简化版ADAR模型
    移除AFB模块，保留核心的对齐和精炼功能
    """
    def __init__(
        self,
        config: MultiModalBartConfig,
        args,
        bart_model: str,
        tokenizer: BartTokenizer,
        label_ids,
        simple_adar_config: SimpleADARModelConfig
    ):
        super().__init__(config)
        self.config = config
        self.args = args
        self.mydevice = args.device
        self.simple_adar_config = simple_adar_config

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

        # ==================== Simple ADAR新组件 ====================

        # 1. OT-based Alignment Module (保留)
        self.ot_alignment = CoarseGrainedAlignment(
            embed_dim=simple_adar_config.embed_dim,
            num_heads=simple_adar_config.ot_num_heads,
            epsilon=simple_adar_config.ot_epsilon,
            max_iter=simple_adar_config.ot_max_iter
        )

        # 2. ADRM Refinement Module (保留)
        self.adrm_refinement = ADRMRefinement(
            embed_dim=simple_adar_config.embed_dim,
            num_heads=simple_adar_config.adrm_num_heads,
            dropout=simple_adar_config.adrm_dropout,
            visual_len=simple_adar_config.visual_len,
            textual_len=simple_adar_config.textual_len
        )

        # 3. Feature Fusion Module (保留)
        if simple_adar_config.fusion_method == 'weighted_sum':
            self.feature_fusion = FeatureFusion(
                embed_dim=simple_adar_config.embed_dim,
                visual_len=simple_adar_config.visual_len,
                textual_len=simple_adar_config.textual_len,
                dropout=simple_adar_config.fusion_dropout
            )
        else:
            self.feature_fusion = FeatureFusionV2(
                embed_dim=simple_adar_config.embed_dim,
                visual_len=simple_adar_config.visual_len,
                textual_len=simple_adar_config.textual_len,
                dropout=simple_adar_config.fusion_dropout,
                fusion_method=simple_adar_config.fusion_method
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
        # 复用ADAR_final.py中的Attention类定义
        from src.model.adar_final import Attention
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
        从[15, 768]扩展到[51, 768]

        Args:
            aligned_features: 对齐特征 [B, textual_len, embed_dim]
            target_dim: 目标维度 (visual_len)

        Returns:
            expanded_features: 扩展特征 [B, target_dim, embed_dim]
        """
        batch_size, _, embed_dim = aligned_features.shape

        # 方法1: 简单重复
        expanded = aligned_features.repeat(1, target_dim // self.simple_adar_config.textual_len + 1, 1)
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
        """准备解码器状态 - 简化版本（无AFB）"""

        # ==================== Multimodal Encoding ====================
        # Multi-modal encoding
        encoder_output = self.multimodal_encoder(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        encoder_outputs = encoder_output.last_hidden_state

        # ==================== Feature Separation ====================
        # Step 1: 分离视觉和文本特征
        visual_features = encoder_outputs[:, :self.simple_adar_config.visual_len, :]  # [B, 51, 768]
        textual_features = encoder_outputs[:, self.simple_adar_config.visual_len:, :]  # [B, 15, 768]

        # ==================== OT Alignment ====================
        # Step 2: 粗粒度OT对齐 (视觉→文本空间)
        aligned_visual = self.ot_alignment(visual_features, textual_features)  # [B, 15, 768]

        # ==================== Feature Concatenation ====================
        # Step 3: 特征拼接 (扩展对齐特征到51维，然后与文本特征拼接)
        expanded_aligned = self.expand_aligned_features(aligned_visual, target_dim=self.simple_adar_config.visual_len)
        concatenated = torch.cat([expanded_aligned, textual_features], dim=1)  # [B, 66, 768]

        # ==================== ADRM Refinement ====================
        # Step 4: ADRM精炼 (2层跨模态注意力)
        refined_features = self.adrm_refinement(concatenated)  # [B, 66, 768]

        # ==================== Feature Fusion ====================
        # Step 5: 特征融合 (对齐特征与精炼特征)
        fused_features = self.feature_fusion(aligned_visual, refined_features)  # [B, 66, 768]

        # ==================== Decode Preparation ====================
        # Prepare for decoder
        noun_embed = self.get_noun_embed(fused_features, noun_mask)

        # Create spans from fused features
        spans = fused_features

        # Apply noun attention (maintain AoM logic)
        noun_attention = self.noun_linear(noun_embed)
        noun_attention = torch.tanh(noun_attention)
        multi = self.multi_linear(fused_features)
        multi = torch.tanh(multi)
        noun_attention = noun_attention.unsqueeze(2).repeat(1, 1, multi.size(2), 1)
        multi = multi.unsqueeze(1).repeat(1, noun_embed.size(1), 1, 1)
        alpha = torch.cat([noun_attention, multi], dim=-1)
        alpha = self.att_linear(alpha).squeeze(-1)
        alpha = torch.softmax(alpha, dim=-1)

        # Weighted sum of noun embeddings
        noun_embed = noun_embed * alpha.unsqueeze(-1)
        noun_embed = noun_embed.sum(dim=1)

        # Apply sentiment GCN if enabled
        if hasattr(self.args, 'gcn_on') and self.args.gcn_on:
            noun_embed = noun_embed + self.senti_linear(noun_embed)
            noun_embed = self.senti_gcn(noun_embed)

        # Apply GAT
        gat_output = self.gat(noun_embed)
        gat_output = torch.tanh(gat_output)
        gat_output = self.gat_linear(gat_output)
        noun_embed = noun_embed + gat_output

        return {
            'spans': spans,
            'noun_embed': noun_embed,
            'fused_features': fused_features,
            'aligned_visual': aligned_visual,
            'visual_features': visual_features,
            'textual_features': textual_features
        }

    def predict(
        self,
        input_ids,
        image_features,
        sentiment_value,
        noun_mask,
        attention_mask=None,
        dependency_matrix=None,
        aesc_infos=None
    ):
        """预测方法 - 简化版本"""

        # Prepare state
        state = self.prepare_state(
            input_ids=input_ids,
            image_features=image_features,
            noun_mask=noun_mask,
            attention_mask=attention_mask,
            dependency_matrix=dependency_matrix,
            sentiment_value=sentiment_value
        )

        # Decode
        spans = state['spans']
        noun_embed = state['noun_embed']

        # Apply decoder
        spans = self.decoder(spans, state, sentiment_value)

        return spans

    def forward(
        self,
        input_ids,
        image_features,
        sentiment_value,
        noun_mask,
        attention_mask=None,
        dependency_matrix=None,
        aesc_infos=None
    ):
        """前向传播 - 简化版本"""

        # Prepare state
        state = self.prepare_state(
            input_ids=input_ids,
            image_features=image_features,
            noun_mask=noun_mask,
            attention_mask=attention_mask,
            dependency_matrix=dependency_matrix,
            sentiment_value=sentiment_value
        )

        # Decode and compute loss
        spans = state['spans']
        noun_embed = state['noun_embed']

        # Apply decoder
        logits = self.decoder(spans, state, sentiment_value)

        # Compute loss
        from src.model.modules import Span_loss
        span_loss_fct = Span_loss()
        loss = span_loss_fct(spans[:, 1:], logits, aesc_infos['span_mask'][:, 1:])

        return loss


def create_simple_adar_model(
    config,
    args,
    bart_model='facebook/bart-base',
    tokenizer=None,
    label_ids=None,
    simple_adar_config=None
):
    """
    Factory function to create SimpleADARModel

    Args:
        config: MultiModalBartConfig
        args: 训练参数
        bart_model: BART预训练模型名称
        tokenizer: 分词器
        label_ids: 标签ID列表
        simple_adar_config: SimpleADARModelConfig对象，如果为None则使用默认配置

    Returns:
        SimpleADARModel: 简化版ADAR模型实例
    """
    if simple_adar_config is None:
        simple_adar_config = SimpleADARModelConfig()

    return SimpleADARModel(
        config=config,
        args=args,
        bart_model=bart_model,
        tokenizer=tokenizer,
        label_ids=label_ids,
        simple_adar_config=simple_adar_config
    )


def test_simple_adar_model():
    """测试简化版ADAR模型"""
    import argparse
    from src.model.config import MultiModalBartConfig
    from transformers import BartTokenizer

    # 创建虚拟配置
    config = MultiModalBartConfig.from_pretrained('facebook/bart-base')

    # 创建虚拟参数
    args = argparse.Namespace()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.gcn_on = False
    args.gcn_dropout = 0.1

    # 创建配置
    simple_config = SimpleADARModelConfig(
        embed_dim=768,
        visual_len=51,
        textual_len=15,
        ot_epsilon=0.01,  # 使用优化参数
        ot_max_iter=500,
        adrm_num_heads=8,
        fusion_method='weighted_sum'
    )

    # 创建虚拟分词器和标签
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    label_ids = list(range(100))

    # 创建模型
    model = create_simple_adar_model(
        config=config,
        args=args,
        bart_model='facebook/bart-base',
        tokenizer=tokenizer,
        label_ids=label_ids,
        simple_adar_config=simple_config
    )

    print("✅ SimpleADARModel 创建成功!")
    print(f"   模型组件:")
    print(f"   - OT对齐: {type(model.ot_alignment).__name__}")
    print(f"   - ADRM精炼: {type(model.adrm_refinement).__name__}")
    print(f"   - 特征融合: {type(model.feature_fusion).__name__}")
    print(f"   - AFB模块: ❌ (已移除)")

    # 创建虚拟输入测试
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 66))
    image_features = [torch.randn(batch_size, 49, 2048) for _ in range(3)]
    sentiment_value = torch.randint(0, 3, (batch_size,))
    noun_mask = torch.ones(batch_size, 15)

    aesc_infos = {
        'spans': torch.randint(0, 10, (batch_size, 20)),
        'labels': torch.randint(0, 100, (batch_size, 20)),
        'span_mask': torch.ones(batch_size, 20)
    }

    # 测试前向传播
    with torch.no_grad():
        loss = model(
            input_ids=input_ids,
            image_features=image_features,
            sentiment_value=sentiment_value,
            noun_mask=noun_mask,
            aesc_infos=aesc_infos
        )

    print(f"   前向传播测试: loss = {loss.item():.4f}")
    print("\n🎉 简化版ADAR模型测试通过!")


if __name__ == '__main__':
    test_simple_adar_model()
