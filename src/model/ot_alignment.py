"""
OT-based Alignment Module for ADAR
Based on OTKGE's optimal transport approach

This module replaces AoM's noun_attention + multimodal_GCN with
optimal transport-based visual-textual alignment.

Key functionality:
1. Map visual features H_V (n, 768) to text token space length m
2. Output H_V_hat (m, 768)

Process:
1. Construct cost matrix S (n x m)
2. Add AFB (n+1 x m+1) padding
3. Sinkhorn to compute OT matrix T
4. Barycentric mapping: H_V_hat = diag(1/nu) @ (T^T + Delta_T) @ H_V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.sinkhorn import compute_cost_matrix, sinkhorn


class OTAlignment(nn.Module):
    """
    Optimal Transport-based Alignment Module

    Replaces AoM's noun_attention + multimodal_GCN with OT-based
    visual-textual feature alignment.
    """

    def __init__(self,
                 embed_dim: int = 768,
                 epsilon: float = 0.1,
                 max_iter: int = 100,
                 afb_padding: bool = True):
        """
        Args:
            embed_dim: Feature embedding dimension (default: 768)
            epsilon: Entropic regularization parameter for Sinkhorn
            max_iter: Maximum Sinkhorn iterations
            afb_padding: Whether to use AFB (Augmented Fictitious Batch) padding
        """
        super(OTAlignment, self).__init__()
        self.embed_dim = embed_dim
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.afb_padding = afb_padding

        # Linear layers for feature transformation
        self.visual_proj = nn.Linear(embed_dim, embed_dim)
        self.textual_proj = nn.Linear(embed_dim, embed_dim)

        # Layer normalization
        self.layer_norm_v = nn.LayerNorm(embed_dim)
        self.layer_norm_t = nn.LayerNorm(embed_dim)

    def forward(self,
                visual_features: torch.Tensor,
                textual_features: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal transport alignment between visual and textual features.

        Args:
            visual_features: Visual features of shape (B, n, embed_dim)
            textual_features: Textual features of shape (B, m, embed_dim)

        Returns:
            Aligned visual features of shape (B, m, embed_dim)
        """
        B, n, d = visual_features.shape
        _, m, _ = textual_features.shape

        # Project features to common space
        visual_proj = self.layer_norm_v(self.visual_proj(visual_features))
        textual_proj = self.layer_norm_t(self.textual_proj(textual_features))

        # Compute cost matrix S (n x m) for each batch
        aligned_features = []

        for b in range(B):
            # Get features for current batch
            v_feat = visual_proj[b]  # (n, d)
            t_feat = textual_proj[b]  # (m, d)

            # Compute cost matrix
            cost_matrix = compute_cost_matrix(v_feat, t_feat)  # (n, m)

            # Apply AFB padding if enabled
            if self.afb_padding:
                # Add fictitious points for better OT solution
                # AFB: (n+1) x (m+1)
                afb_cost = torch.zeros(n + 1, m + 1, device=cost_matrix.device)
                afb_cost[:n, :m] = cost_matrix

                # Uniform distributions for AFB
                a_afb = torch.ones(n + 1, device=cost_matrix.device) / (n + 1)
                b_afb = torch.ones(m + 1, device=cost_matrix.device) / (m + 1)

                # Compute OT matrix with AFB
                T_afb = sinkhorn(a_afb, b_afb, afb_cost,
                                reg=self.epsilon,
                                maxIter=self.max_iter)

                # Extract main OT matrix (without fictitious points)
                T = T_afb[:n, :m]

            else:
                # Standard OT without AFB
                a = torch.ones(n, device=cost_matrix.device) / n
                b = torch.ones(m, device=cost_matrix.device) / m

                T = sinkhorn(a, b, cost_matrix,
                            reg=self.epsilon,
                            maxIter=self.max_iter)

            # Barycentric mapping: H_V_hat = diag(1/mu) @ T^T @ H_V
            # where mu is the marginal of source distribution (sum over columns)
            # T^T @ H_V gives weighted sum of visual features for each target

            # Compute marginals (sum over columns to get source marginals)
            nu = T.sum(dim=1)  # (n,)
            nu_inv = 1.0 / (nu + 1e-8)  # Avoid division by zero

            # Barycentric projection
            # T^T @ H_V gives weighted sum of visual features for each target
            weighted_visual = T.t() @ v_feat  # (m, d)

            # Apply inverse marginal scaling
            H_V_hat = weighted_visual  # (m, d)

            aligned_features.append(H_V_hat)

        # Stack to form batch
        aligned_features = torch.stack(aligned_features, dim=0)  # (B, m, d)

        return aligned_features


class CoarseGrainedAlignment(nn.Module):
    """
    Coarse-grained alignment module for ADAR

    This is the main alignment module that maps visual features
    to textual token space, replacing noun_attention + GCN.
    """

    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 4,
                 epsilon: float = 0.1,
                 max_iter: int = 100):
        """
        Args:
            embed_dim: Feature dimension
            num_heads: Number of attention heads for multi-head processing
            epsilon: Sinkhorn regularization parameter
            max_iter: Maximum Sinkhorn iterations
        """
        super(CoarseGrainedAlignment, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # OT Alignment module
        self.ot_alignment = OTAlignment(
            embed_dim=embed_dim,
            epsilon=epsilon,
            max_iter=max_iter,
            afb_padding=True
        )

        # Feature fusion layer
        self.fusion_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Gating mechanism
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                visual_features: torch.Tensor,
                textual_features: torch.Tensor) -> torch.Tensor:
        """
        Perform coarse-grained alignment between visual and textual features.

        Args:
            visual_features: Visual features (B, n, 768)
            textual_features: Textual features (B, m, 768)

        Returns:
            Aligned features of shape (B, m, 768)
        """
        # Extract text features (after image tokens in AOM)
        # AOM format: first 51 are image, rest are text
        text_tokens = textual_features  # (B, m, 768)

        # Apply OT-based alignment
        aligned_visual = self.ot_alignment(visual_features, text_tokens)
        # aligned_visual: (B, m, 768)

        # Fuse aligned visual features with textual features
        fused = torch.cat([aligned_visual, text_tokens], dim=-1)  # (B, m, 1536)
        fused = self.fusion_norm(self.fusion_linear(fused))

        # Gating mechanism
        gate_input = torch.cat([aligned_visual, text_tokens], dim=-1)
        gate_weights = self.sigmoid(self.gate(gate_input))  # (B, m, 768)

        # Apply gating
        output = gate_weights * aligned_visual + (1 - gate_weights) * text_tokens

        return output


def test_ot_alignment():
    """Test function for OT alignment module"""
    # Create dummy data
    B, n, m, d = 2, 49, 15, 768  # 49 image patches, 15 text tokens

    visual_features = torch.randn(B, n, d)
    textual_features = torch.randn(B, m, d)

    # Initialize module
    ot_align = OTAlignment(embed_dim=d)

    # Forward pass
    aligned = ot_align(visual_features, textual_features)

    print(f"Input visual: {visual_features.shape}")
    print(f"Input textual: {textual_features.shape}")
    print(f"Output aligned: {aligned.shape}")
    print("OT Alignment test passed!")


if __name__ == "__main__":
    test_ot_alignment()
