"""
ADRM Refinement Module
跨模态注意力精炼模块

功能: 通过2层跨模态注意力机制精炼多模态特征
输入: [B, n+m, 768] (拼接特征)
输出: [B, n+m, 768] (精炼特征)

实现原理:
1. 层1: 视觉→文本注意力 (n→m)
2. 层2: 文本→视觉注意力 (m→n)
3. 多头注意力机制
4. 残差连接 + 层归一化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    实现从源模态到目标模态的注意力
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        source_len: int = 51,
        target_len: int = 15
    ):
        """
        Args:
            embed_dim: 特征嵌入维度
            num_heads: 注意力头数
            dropout: dropout率
            source_len: 源模态长度
            target_len: 目标模态长度
        """
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.source_len = source_len
        self.target_len = target_len

        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        # 线性变换层，用于将输入映射到Q, K, V空间
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        跨模态注意力前向传播

        Args:
            source_features: 源模态特征 [B, source_len, embed_dim]
            target_features: 目标模态特征 [B, target_len, embed_dim]

        Returns:
            输出特征 [B, target_len, embed_dim]
        """
        batch_size = source_features.size(0)

        # 线性变换得到Q, K, V
        query = self.query_proj(target_features)  # [B, target_len, embed_dim]
        key = self.key_proj(source_features)      # [B, source_len, embed_dim]
        value = self.value_proj(source_features)  # [B, source_len, embed_dim]

        # 将embed_dim划分为num_heads个头
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # attention_scores: [B, num_heads, target_len, source_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # 应用注意力权重到value
        # output: [B, num_heads, target_len, head_dim]
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )

        # 输出线性变换
        output = self.output_proj(output)

        # 残差连接
        output = output + target_features

        # 层归一化
        output = self.layer_norm(output)

        return output


class ADRMRefinement(nn.Module):
    """
    ADRM (Aspect Detection with Attention Reorganization Module) 精炼模块
    通过2层跨模态注意力机制精炼多模态特征
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        visual_len: int = 51,
        textual_len: int = 15
    ):
        """
        Args:
            embed_dim: 特征嵌入维度
            num_heads: 注意力头数
            dropout: dropout率
            visual_len: 视觉特征长度
            textual_len: 文本特征长度
        """
        super(ADRMRefinement, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.visual_len = visual_len
        self.textual_len = textual_len

        # 第一层: 视觉→文本注意力 (n→m)
        self.visual_to_textual_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            source_len=visual_len,
            target_len=textual_len
        )

        # 第二层: 文本→视觉注意力 (m→n)
        self.textual_to_visual_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            source_len=textual_len,
            target_len=visual_len
        )

        # 特征融合层
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 拼接特征 [B, visual_len + textual_len, embed_dim]

        Returns:
            精炼特征 [B, visual_len + textual_len, embed_dim]
        """
        batch_size = features.size(0)

        # 分离视觉和文本特征
        visual_features = features[:, :self.visual_len, :]  # [B, visual_len, embed_dim]
        textual_features = features[:, self.visual_len:, :]  # [B, textual_len, embed_dim]

        # 第一层: 视觉→文本注意力
        # 使用视觉特征作为查询，文本特征作为键值
        textual_refined = self.visual_to_textual_attention(
            source_features=visual_features,
            target_features=textual_features
        )  # [B, textual_len, embed_dim]

        # 第二层: 文本→视觉注意力
        # 使用文本特征作为查询，视觉特征作为键值
        visual_refined = self.textual_to_visual_attention(
            source_features=textual_refined,
            target_features=visual_features
        )  # [B, visual_len, embed_dim]

        # 融合精炼后的特征
        # 方法1: 直接拼接 (simple concatenation)
        refined_visual = self.fusion_proj(
            torch.cat([visual_features, visual_refined], dim=-1)
        )
        refined_visual = self.fusion_norm(refined_visual)

        refined_textual = self.fusion_proj(
            torch.cat([textual_features, textual_refined], dim=-1)
        )
        refined_textual = self.fusion_norm(refined_textual)

        # 拼接精炼后的视觉和文本特征
        refined_features = torch.cat(
            [refined_visual, refined_textual],
            dim=1
        )  # [B, visual_len + textual_len, embed_dim]

        # 添加dropout
        refined_features = self.dropout_layer(refined_features)

        # 残差连接
        refined_features = refined_features + features

        return refined_features


def test_adrm_refinement():
    """测试ADRM精炼模块"""
    # 创建虚拟数据
    batch_size = 2
    visual_len = 51
    textual_len = 15
    embed_dim = 768

    # 创建虚拟输入
    features = torch.randn(batch_size, visual_len + textual_len, embed_dim)
    print(f"输入特征: {features.shape}")

    # 创建ADRM精炼模块
    adrm = ADRMRefinement(
        embed_dim=embed_dim,
        num_heads=8,
        dropout=0.1,
        visual_len=visual_len,
        textual_len=textual_len
    )

    # 前向传播
    with torch.no_grad():
        output = adrm(features)

    print(f"输出特征: {output.shape}")
    print(f"输出类型: {output.dtype}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")

    # 测试跨模态注意力
    visual_features = features[:, :visual_len, :]
    textual_features = features[:, visual_len:, :]

    # 测试视觉→文本注意力
    cross_attention = CrossModalAttention(
        embed_dim=embed_dim,
        num_heads=8,
        source_len=visual_len,
        target_len=textual_len
    )

    with torch.no_grad():
        textual_attended = cross_attention(
            source_features=visual_features,
            target_features=textual_features
        )

    print(f"\n跨模态注意力测试:")
    print(f"输入文本特征: {textual_features.shape}")
    print(f"输出文本特征: {textual_attended.shape}")

    print("\n✅ ADRM精炼模块测试通过!")


if __name__ == "__main__":
    test_adrm_refinement()