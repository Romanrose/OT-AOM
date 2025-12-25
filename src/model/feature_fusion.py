"""
Feature Fusion Module
特征融合模块

功能: 加权融合对齐特征与精炼特征
输入:
  - aligned_features: [B, m, 768] (OT对齐视觉特征)
  - refined_features: [B, n+m, 768] (ADRM精炼特征)
输出: [B, n+m, 768] (融合特征)

实现原理:
1. 分离精炼特征: 视觉[n] + 文本[m]
2. 加权融合: H_fused = γ * aligned_text + (1-γ) * refined_text
3. γ为可学习权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    特征融合模块
    通过可学习权重融合OT对齐特征与ADRM精炼特征
    """

    def __init__(
        self,
        embed_dim: int = 768,
        visual_len: int = 51,
        textual_len: int = 15,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 特征嵌入维度
            visual_len: 视觉特征长度
            textual_len: 文本特征长度
            dropout: dropout率
        """
        super(FeatureFusion, self).__init__()
        self.embed_dim = embed_dim
        self.visual_len = visual_len
        self.textual_len = textual_len

        # 可学习的融合权重 γ
        # γ 控制对齐特征与精炼特征的融合比例
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)

        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        aligned_features: torch.Tensor,
        refined_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            aligned_features: OT对齐的视觉特征 [B, textual_len, embed_dim]
            refined_features: ADRM精炼特征 [B, visual_len + textual_len, embed_dim]

        Returns:
            融合特征 [B, visual_len + textual_len, embed_dim]
        """
        batch_size = aligned_features.size(0)

        # 分离精炼特征为视觉和文本部分
        refined_visual = refined_features[:, :self.visual_len, :]  # [B, visual_len, embed_dim]
        refined_textual = refined_features[:, self.visual_len:, :]  # [B, textual_len, embed_dim]

        # 将对齐特征扩展到视觉特征维度 (以便与视觉精炼特征融合)
        # 方法: 复制对齐特征到所有视觉位置
        expanded_aligned = aligned_features.repeat(1, self.visual_len // self.textual_len + 1, 1)
        expanded_aligned = expanded_aligned[:, :self.visual_len, :]  # [B, visual_len, embed_dim]

        # 加权融合视觉特征
        # H_visual_fused = γ * expanded_aligned + (1-γ) * refined_visual
        gamma = torch.sigmoid(self.fusion_weight)  # 使用sigmoid确保权重在[0,1]范围内
        visual_fused = gamma * expanded_aligned + (1 - gamma) * refined_visual

        # 加权融合文本特征
        # H_text_fused = γ * aligned_features + (1-γ) * refined_textual
        textual_fused = gamma * aligned_features + (1 - gamma) * refined_textual

        # 融合特征变换
        visual_fused = self.feature_transform(
            torch.cat([expanded_aligned, visual_fused], dim=-1)
        )
        visual_fused = self.layer_norm(visual_fused)

        textual_fused = self.feature_transform(
            torch.cat([aligned_features, textual_fused], dim=-1)
        )
        textual_fused = self.layer_norm(textual_fused)

        # 应用dropout
        visual_fused = self.dropout(visual_fused)
        textual_fused = self.dropout(textual_fused)

        # 拼接融合后的视觉和文本特征
        fused_features = torch.cat([visual_fused, textual_fused], dim=1)

        # 残差连接
        fused_features = fused_features + refined_features

        return fused_features


class FeatureFusionV2(nn.Module):
    """
    特征融合模块V2 - 改进版本
    使用更复杂的融合策略
    """

    def __init__(
        self,
        embed_dim: int = 768,
        visual_len: int = 51,
        textual_len: int = 15,
        dropout: float = 0.1,
        fusion_method: str = 'weighted_sum'
    ):
        """
        Args:
            embed_dim: 特征嵌入维度
            visual_len: 视觉特征长度
            textual_len: 文本特征长度
            dropout: dropout率
            fusion_method: 融合方法 ('weighted_sum', 'gated', 'attention')
        """
        super(FeatureFusionV2, self).__init__()
        self.embed_dim = embed_dim
        self.visual_len = visual_len
        self.textual_len = textual_len
        self.fusion_method = fusion_method

        if fusion_method == 'weighted_sum':
            # 可学习的融合权重
            self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)
        elif fusion_method == 'gated':
            # 门控融合机制
            self.gate = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif fusion_method == 'attention':
            # 注意力融合机制
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(embed_dim)

        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        aligned_features: torch.Tensor,
        refined_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            aligned_features: OT对齐的视觉特征 [B, textual_len, embed_dim]
            refined_features: ADRM精炼特征 [B, visual_len + textual_len, embed_dim]

        Returns:
            融合特征 [B, visual_len + textual_len, embed_dim]
        """
        batch_size = aligned_features.size(0)

        # 分离精炼特征为视觉和文本部分
        refined_visual = refined_features[:, :self.visual_len, :]
        refined_textual = refined_features[:, self.visual_len:, :]

        # 将对齐特征扩展到视觉特征维度
        expanded_aligned = aligned_features.repeat(1, self.visual_len // self.textual_len + 1, 1)
        expanded_aligned = expanded_aligned[:, :self.visual_len, :]

        # 根据融合方法进行特征融合
        if self.fusion_method == 'weighted_sum':
            # 加权融合
            gamma = torch.sigmoid(self.fusion_weight)
            visual_fused = gamma * expanded_aligned + (1 - gamma) * refined_visual
            textual_fused = gamma * aligned_features + (1 - gamma) * refined_textual
        elif self.fusion_method == 'gated':
            # 门控融合
            visual_cat = torch.cat([expanded_aligned, refined_visual], dim=-1)
            gate_values = torch.sigmoid(self.gate(visual_cat))
            visual_fused = gate_values * expanded_aligned + (1 - gate_values) * refined_visual

            textual_cat = torch.cat([aligned_features, refined_textual], dim=-1)
            gate_values = torch.sigmoid(self.gate(textual_cat))
            textual_fused = gate_values * aligned_features + (1 - gate_values) * refined_textual
        elif self.fusion_method == 'attention':
            # 注意力融合
            # 视觉特征: 使用对齐特征作为query，精炼特征作为key和value
            visual_fused, _ = self.attention(
                query=expanded_aligned,
                key=refined_visual,
                value=refined_visual
            )
            visual_fused = self.attention_norm(visual_fused + expanded_aligned)

            # 文本特征: 使用对齐特征作为query，精炼特征作为key和value
            textual_fused, _ = self.attention(
                query=aligned_features,
                key=refined_textual,
                value=refined_textual
            )
            textual_fused = self.attention_norm(textual_fused + aligned_features)
        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")

        # 特征变换
        visual_fused = self.feature_transform(
            torch.cat([expanded_aligned, visual_fused], dim=-1)
        )
        visual_fused = self.layer_norm(visual_fused)

        textual_fused = self.feature_transform(
            torch.cat([aligned_features, textual_fused], dim=-1)
        )
        textual_fused = self.layer_norm(textual_fused)

        # 应用dropout
        visual_fused = self.dropout(visual_fused)
        textual_fused = self.dropout(textual_fused)

        # 拼接融合后的视觉和文本特征
        fused_features = torch.cat([visual_fused, textual_fused], dim=1)

        # 残差连接
        fused_features = fused_features + refined_features

        return fused_features


def test_feature_fusion():
    """测试特征融合模块"""
    # 创建虚拟数据
    batch_size = 2
    visual_len = 51
    textual_len = 15
    embed_dim = 768

    # 创建虚拟输入
    aligned_features = torch.randn(batch_size, textual_len, embed_dim)
    refined_features = torch.randn(batch_size, visual_len + textual_len, embed_dim)

    print(f"对齐特征: {aligned_features.shape}")
    print(f"精炼特征: {refined_features.shape}")

    # 创建特征融合模块
    feature_fusion = FeatureFusion(
        embed_dim=embed_dim,
        visual_len=visual_len,
        textual_len=textual_len,
        dropout=0.1
    )

    # 前向传播
    with torch.no_grad():
        output = feature_fusion(aligned_features, refined_features)

    print(f"输出特征: {output.shape}")
    print(f"输出类型: {output.dtype}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")

    # 测试特征融合V2 (加权融合)
    feature_fusion_v2 = FeatureFusionV2(
        embed_dim=embed_dim,
        visual_len=visual_len,
        textual_len=textual_len,
        dropout=0.1,
        fusion_method='weighted_sum'
    )

    with torch.no_grad():
        output_v2 = feature_fusion_v2(aligned_features, refined_features)

    print(f"\n特征融合V2输出: {output_v2.shape}")
    print(f"融合权重: {torch.sigmoid(feature_fusion_v2.fusion_weight).item():.4f}")

    # 测试特征融合V2 (门控融合)
    feature_fusion_v2_gate = FeatureFusionV2(
        embed_dim=embed_dim,
        visual_len=visual_len,
        textual_len=textual_len,
        dropout=0.1,
        fusion_method='gated'
    )

    with torch.no_grad():
        output_v2_gate = feature_fusion_v2_gate(aligned_features, refined_features)

    print(f"\n特征融合V2(门控)输出: {output_v2_gate.shape}")

    print("\n✅ 特征融合模块测试通过!")


if __name__ == "__main__":
    test_feature_fusion()