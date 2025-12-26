"""
AFB (Adaptive Filtering Bin) Module
自适应过滤仓模块

功能: 过滤视觉特征中的噪声，保留有用信息
输入: [B, n, 768] (n=51视觉token)
输出: [B, n, 768] (过滤后视觉特征)

实现原理:
1. 计算注意力权重: α = sigmoid(Linear(H))
2. 过滤: H_filtered = α ⊙ H
3. 残差连接: H_out = H + β * H_filtered
4. β为可学习权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFBModule(nn.Module):
    """
    Adaptive Filtering Bin (AFB) Module

    自适应过滤仓，用于过滤视觉特征中的噪声
    """

    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        """
        Args:
            embed_dim: Feature embedding dimension (default: 768)
            dropout: Dropout rate for regularization
        """
        super(AFBModule, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        # 注意力权重计算层
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 可学习的残差权重
        self.beta = nn.Parameter(torch.ones(1))

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of AFB module

        Args:
            visual_features: Visual features of shape [B, n, 768]
                B: batch size
                n: number of visual tokens (51 for AoM/ADAR)
                768: feature dimension

        Returns:
            Filtered visual features of shape [B, n, 768]
        """
        B, n, d = visual_features.shape

        # 1. 检查并调整维度顺序
        # 期望的形状: [B, visual_len, embed_dim] = [B, 51, 768]
        # 如果维度顺序错误，交换维度
        if d == 768 and n == 51:
            # 正确形状: [B, 51, 768]，保持不变
            pass
        elif n == 768 and d == 51:
            # 实际形状是 [B, embed_dim, visual_len]，需要交换
            visual_features = visual_features.transpose(1, 2)  # 转换为 [B, 51, 768]
            n, d = d, n  # 交换n和d
        else:
            # 未知形状，报错
            raise ValueError(f"Unexpected tensor shape: {visual_features.shape}. Expected [B, 51, 768] or [B, 768, 51]")

        # 确保张量是连续的
        if not visual_features.is_contiguous():
            visual_features = visual_features.contiguous()

        # 验证张量维度
        assert visual_features.shape == (B, n, d), f"Expected shape ({B}, {n}, {d}), got {visual_features.shape}"

        # 将特征重塑为 [B*n, 768] 以便计算注意力
        # 使用reshape代替view以处理非连续张量
        features_flat = visual_features.reshape(-1, d)  # [B*n, 768]

        # 计算每个位置的注意力权重
        attention_weights = self.attention_weights(features_flat)  # [B*n, 1]
        attention_weights = attention_weights.reshape(B, n, 1)  # [B, n, 1]

        # 2. 特征变换
        transformed = self.feature_transform(visual_features)  # [B, n, 768]

        # 3. 应用注意力权重
        filtered = attention_weights * transformed  # [B, n, 768]

        # 4. 残差连接
        # H_out = H + β * H_filtered
        output = visual_features + self.beta * filtered

        # 5. 层归一化
        output = self.layer_norm(output)

        return output


class AFBModuleV2(nn.Module):
    """
    AFB Module Version 2 - Enhanced with multi-head attention

    改进版本的AFB模块，添加了多头注意力机制
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Feature embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AFBModuleV2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # 注意力权重计算
        self.attention_weights = nn.Linear(embed_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 可学习残差权重
        self.beta = nn.Parameter(torch.ones(1))

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention

        Args:
            visual_features: [B, n, 768]

        Returns:
            Filtered features: [B, n, 768]
        """
        B, n, d = visual_features.shape

        # 1. 计算多头注意力权重
        # [B, n, num_heads]
        attention_weights = self.attention_weights(visual_features)

        # 应用softmax得到归一化权重
        attention_weights = self.softmax(attention_weights)

        # 2. 特征变换
        transformed = self.feature_transform(visual_features)  # [B, n, 768]

        # 3. 应用多头注意力
        # 扩展权重维度以匹配特征维度
        attention_weights = attention_weights.unsqueeze(-1)  # [B, n, num_heads, 1]

        # 计算加权和
        filtered = torch.sum(attention_weights * transformed.unsqueeze(2), dim=2)  # [B, n, 768]

        # 4. 残差连接
        output = visual_features + self.beta * filtered

        # 5. 层归一化
        output = self.layer_norm1(output)

        return output


def test_afb_module():
    """Test function for AFB module"""
    # Create dummy data
    B, n, d = 2, 51, 768  # batch size, visual tokens, feature dim

    visual_features = torch.randn(B, n, d)
    print(f"Input visual features: {visual_features.shape}")

    # Test AFBModule
    afb = AFBModule(embed_dim=d)
    output = afb(visual_features)
    print(f"AFB output: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test AFBModuleV2
    afb_v2 = AFBModuleV2(embed_dim=d, num_heads=8)
    output_v2 = afb_v2(visual_features)
    print(f"AFB V2 output: {output_v2.shape}")

    # Check if attention weights are reasonable
    with torch.no_grad():
        attention_weights = afb.attention_weights(visual_features.view(-1, d))
        attention_weights = attention_weights.view(B, n, 1)
        print(f"Attention weights range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
        print(f"Mean attention weight: {attention_weights.mean():.4f}")

    print("\n✅ AFB Module test passed!")


if __name__ == "__main__":
    test_afb_module()
