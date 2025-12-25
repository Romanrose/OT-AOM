"""
ADAR Final Model
完整的ADAR模型实现，集成所有新组件

基于新数据流的完整ADAR模型:
输入 → ResNet → [B, 2048, 7, 7] → [B, 49, 2048] → ImageEmbedding → [B, 51, 768]
    → 文本嵌入 → [B, 15, 768] → 多模态编码 → [B, 66, 768]
    → 分离: 视觉[B, 51, 768] + 文本[B, 15, 768]
    → 粗粒度OT对齐 → [B, 15, 768]  (对齐视觉到文本空间)
    → AFB过滤 → [B, 51, 768]      (过滤视觉噪声，保留维度)
    → 特征拼接 → [B, 66, 768]    (垂直拼接: 51+15)
    → ADRM精炼 → [B, 66, 768]    (2层跨模态注意力)
    → 特征融合 → [B, 66, 768]    (加权融合)
    → 解码器 → 输出

组件列表:
1. MultiModalBartEncoder: BART多模态编码器
2. CoarseGrainedAlignment: 粗粒度OT对齐模块
3. AFBModule: 自适应过滤仓
4. ADRMRefinement: 2层跨模态注意力精炼模块
5. FeatureFusion: 特征融合层
6. MultiModalBartDecoder_span: BART多模态解码器
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from src.model.config import MultiModalBartConfig
from src.model.modeling_bart import PretrainedBartModel, BartModel
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span
from src.model.ot_alignment import CoarseGrainedAlignment
from src.model.afb import AFBModule, AFBModuleV2
from src.model.adrm_refinement import ADRMRefinement
from src.model.feature_fusion import FeatureFusion, FeatureFusionV2
from transformers import BartTokenizer


class Attention(nn.Module):
    """
    Multi-head attention layer (moved to module level for pickling)
    """
    def __init__(self, num_attention_heads: int, input_size: int, hidden_size: int):
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
        from torch.nn import functional as F
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


class ADARModelConfig:
    """
    ADAR模型配置类
    """
    def __init__(
        self,
        # 基础配置
        embed_dim: int = 768,
        visual_len: int = 51,
        textual_len: int = 15,

        # AFB配置
        afb_dropout: float = 0.1,
        use_afb_v2: bool = False,

        # OT对齐配置
        ot_num_heads: int = 4,
        ot_epsilon: float = 0.01,
        ot_max_iter: int = 100,

        # ADRM精炼配置
        adrm_num_heads: int = 8,
        adrm_dropout: float = 0.1,

        # 特征融合配置
        fusion_method: str = 'weighted_sum',  # 'weighted_sum', 'gated', 'attention'
        fusion_dropout: float = 0.1
    ):
        self.embed_dim = embed_dim
        self.visual_len = visual_len
        self.textual_len = textual_len

        self.afb_dropout = afb_dropout
        self.use_afb_v2 = use_afb_v2

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
            'afb_dropout': self.afb_dropout,
            'use_afb_v2': self.use_afb_v2,
            'ot_num_heads': self.ot_num_heads,
            'ot_epsilon': self.ot_epsilon,
            'ot_max_iter': self.ot_max_iter,
            'adrm_num_heads': self.adrm_num_heads,
            'adrm_dropout': self.adrm_dropout,
            'fusion_method': self.fusion_method,
            'fusion_dropout': self.fusion_dropout
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ADARModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)


class ADARModelFinal(PretrainedBartModel):
    """
    完整的ADAR模型
    集成所有新组件的完整实现
    """
    def __init__(
        self,
        config: MultiModalBartConfig,
        args,
        bart_model: str,
        tokenizer: BartTokenizer,
        label_ids,
        adar_config: ADARModelConfig
    ):
        super().__init__(config)
        self.config = config
        self.args = args
        self.mydevice = args.device



        label_ids = sorted(label_ids)
        self.adar_config = adar_config

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

        # ==================== ADAR新组件 ====================

        # 1. OT-based Alignment Module
        self.ot_alignment = CoarseGrainedAlignment(
            embed_dim=adar_config.embed_dim,
            num_heads=adar_config.ot_num_heads,
            epsilon=adar_config.ot_epsilon,
            max_iter=adar_config.ot_max_iter
        )

        # 2. AFB Module
        if adar_config.use_afb_v2:
            self.afb_module = AFBModuleV2(
                embed_dim=adar_config.embed_dim,
                num_heads=adar_config.adrm_num_heads,
                dropout=adar_config.afb_dropout
            )
        else:
            self.afb_module = AFBModule(
                embed_dim=adar_config.embed_dim,
                dropout=adar_config.afb_dropout
            )

        # 3. ADRM Refinement Module
        self.adrm_refinement = ADRMRefinement(
            embed_dim=adar_config.embed_dim,
            num_heads=adar_config.adrm_num_heads,
            dropout=adar_config.adrm_dropout,
            visual_len=adar_config.visual_len,
            textual_len=adar_config.textual_len
        )

        # 4. Feature Fusion Module
        if adar_config.fusion_method == 'weighted_sum':
            self.feature_fusion = FeatureFusion(
                embed_dim=adar_config.embed_dim,
                visual_len=adar_config.visual_len,
                textual_len=adar_config.textual_len,
                dropout=adar_config.fusion_dropout
            )
        else:
            self.feature_fusion = FeatureFusionV2(
                embed_dim=adar_config.embed_dim,
                visual_len=adar_config.visual_len,
                textual_len=adar_config.textual_len,
                dropout=adar_config.fusion_dropout,
                fusion_method=adar_config.fusion_method
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
        expanded = aligned_features.repeat(1, target_dim // self.adar_config.textual_len + 1, 1)
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


        print('encoder_outputs',encoder_outputs.shape)
        # 步骤1: 分离视觉和文本特征
        visual_features = encoder_outputs[:, :self.adar_config.visual_len, :]  # [B, 51, 768]
        textual_features = encoder_outputs[:, self.adar_config.visual_len:, :]  # [B, 15, 768]
        print('visual_features',visual_features.shape)
        print('textual_features',textual_features.shape)
        # 步骤2: 粗粒度OT对齐 (视觉→文本空间)
        aligned_visual = self.ot_alignment(visual_features, textual_features)  # [B, 15, 768]
        print('aligned_visual',aligned_visual.shape)

        # 步骤3: AFB过滤 (过滤视觉噪声，保留维度)
        filtered_visual = self.afb_module(visual_features)  # [B, 51, 768]
        print('filtered_visual',filtered_visual.shape)
        # 步骤4: 特征拼接
        # 将aligned_visual [15] 扩展到 [51] 然后与 filtered_visual [51] 垂直拼接
        expanded_aligned = self.expand_aligned_features(aligned_visual, target_dim=self.adar_config.visual_len)
        print('expanded_aligned',expanded_aligned.shape)
        concatenated = torch.cat([expanded_aligned, textual_features], dim=1)  # [B, 66, 768]

        # 步骤5: ADRM精炼 (2层跨模态注意力)
        refined_features = self.adrm_refinement(concatenated)  # [B, 66, 768]
        print('refined_features',refined_features.shape)
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
            src_tokens=input_ids[:, self.adar_config.visual_len:],
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


def create_adar_model_final(
    config,
    args,
    bart_model='facebook/bart-base',
    tokenizer=None,
    label_ids=None,
    adar_config=None
):
    """
    Factory function to create ADARModelFinal

    Args:
        config: MultiModalBartConfig
        args: 训练参数
        bart_model: BART预训练模型名称
        tokenizer: 分词器
        label_ids: 标签ID列表
        adar_config: ADARModelConfig对象，如果为None则使用默认配置

    Returns:
        ADARModelFinal: 完整的ADAR模型实例
    """
    if adar_config is None:
        adar_config = ADARModelConfig()

    return ADARModelFinal(
        config=config,
        args=args,
        bart_model=bart_model,
        tokenizer=tokenizer,
        label_ids=label_ids,
        adar_config=adar_config
    )


def test_adar_model_final():
    """测试完整ADAR模型"""
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

    # 创建ADAR配置
    adar_config = ADARModelConfig(
        embed_dim=768,
        visual_len=51,
        textual_len=15,
        afb_dropout=0.1,
        use_afb_v2=False,
        ot_num_heads=4,
        ot_epsilon=0.1,
        ot_max_iter=100,
        adrm_num_heads=8,
        adrm_dropout=0.1,
        fusion_method='weighted_sum',
        fusion_dropout=0.1
    )

    # 创建虚拟分词器和标签
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    label_ids = list(range(10))  # 虚拟标签ID

    # 创建模型
    model = create_adar_model_final(
        config=config,
        args=args,
        bart_model='facebook/bart-base',
        tokenizer=tokenizer,
        label_ids=label_ids,
        adar_config=adar_config
    )

    print(f"模型创建成功: {type(model).__name__}")
    print(f"视觉长度: {model.adar_config.visual_len}")
    print(f"文本长度: {model.adar_config.textual_len}")
    print(f"OT对齐头数: {model.adar_config.ot_num_heads}")
    print(f"ADRM精炼头数: {model.adar_config.adrm_num_heads}")
    print(f"特征融合方法: {model.adar_config.fusion_method}")

    # 创建虚拟输入
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 66))
    image_features = torch.randn(batch_size, 3, 224, 224)
    sentiment_value = torch.randint(0, 10, (batch_size,))
    noun_mask = torch.ones(batch_size, 15)
    attention_mask = torch.ones(batch_size, 66)

    # 创建虚拟aesc_infos
    aesc_infos = {
        'labels': torch.randint(0, 10, (batch_size, 10)),
        'masks': torch.ones(batch_size, 10)
    }

    # 前向传播
    try:
        with torch.no_grad():
            loss = model(
                input_ids=input_ids,
                image_features=image_features,
                sentiment_value=sentiment_value,
                noun_mask=noun_mask,
                attention_mask=attention_mask,
                aesc_infos=aesc_infos
            )
        print(f"\n✅ 前向传播成功，损失: {loss.item():.4f}")
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")


if __name__ == "__main__":
    test_adar_model_final()