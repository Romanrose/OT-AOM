"""
Sinkhorn Algorithm Implementation
Based on OTKGE's sinkhorn_knopp implementation

This module provides the Sinkhorn algorithm for optimal transport
with entropic regularization, adapted for ADAR model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

M_EPS = torch.tensor(1e-16)


def sinkhorn_knopp(a: torch.Tensor,
                   b: torch.Tensor,
                   C: torch.Tensor,
                   reg: float = 1e-1,
                   maxIter: int = 1000,
                   stopThr: float = 1e-9,
                   verbose: bool = False,
                   device: str = 'cuda') -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for optimal transport.

    This is adapted from OTKGE's implementation.

    Args:
        a: Source marginal distribution of shape (na,)
        b: Target marginal distribution of shape (nb,)
        C: Cost matrix of shape (na, nb)
        reg: Entropic regularization parameter (epsilon)
        maxIter: Maximum number of iterations
        stopThr: Stopping threshold
        verbose: Whether to print progress
        device: Device to run on

    Returns:
        Transport matrix P of shape (na, nb)
    """
    device = a.device

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b doesn't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    # Initialize potentials
    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    # Compute kernel matrix K = exp(-reg * C)
    K = torch.exp(-reg * C)

    M_EPS = torch.tensor(1e-16).to(device)

    it = 1
    err = 1

    while err > stopThr and it <= maxIter:
        u_old, v_old = u, v

        # Update v: v = b / (K^T u)
        KTu = torch.matmul(u.to(torch.float32), K.to(torch.float32))
        v = torch.div(b, KTu + M_EPS)

        # Update u: u = a / (K v)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        # Check for numerical errors
        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print(f'Warning: numerical errors at iteration {it}')
            u, v = u_old, v_old
            break

        # Compute error every 10 iterations
        if it % 10 == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()

        if verbose and it % 100 == 0:
            print(f'iteration {it:5d}, constraint error {err:5e}')

        it += 1

    # Final transport matrix P = diag(u) * K * diag(v)
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)

    return P


def compute_cost_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance cost matrix.

    Args:
        X: Source features of shape (n, d)
        Y: Target features of shape (m, d)

    Returns:
        Cost matrix of shape (n, m)
    """
    # C_ij = ||X_i - Y_j||^2
    X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
    Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
    dist_matrix = X_norm + Y_norm.t() - 2 * X @ Y.t()
    return dist_matrix


def sinkhorn(a: torch.Tensor,
             b: torch.Tensor,
             C: torch.Tensor,
             reg: float = 1e-1,
             maxIter: int = 1000,
             stopThr: float = 1e-9,
             verbose: bool = False,
             device: str = 'cuda') -> torch.Tensor:
    """
    Sinkhorn algorithm for optimal transport.

    Main entry point for computing optimal transport matrix.

    Args:
        a: Source marginal distribution
        b: Target marginal distribution
        C: Cost matrix
        reg: Regularization parameter
        maxIter: Maximum iterations
        stopThr: Stopping threshold
        verbose: Verbosity
        device: Device

    Returns:
        Transport matrix
    """
    return sinkhorn_knopp(a, b, C, reg, maxIter, stopThr, verbose, device)


def cal_ot(X: torch.Tensor,
           Y: torch.Tensor,
           epsilon: float = 0.1,
           max_iter: int = 100) -> torch.Tensor:
    """
    Calculate optimal transport matrix between two sets of features.

    Args:
        X: Source features of shape (n, d)
        Y: Target features of shape (m, d)
        epsilon: Entropic regularization parameter
        max_iter: Maximum number of Sinkhorn iterations

    Returns:
        Transport matrix of shape (n, m)
    """
    device = X.device

    # Compute cost matrix
    C = compute_cost_matrix(X, Y)

    # Create uniform distributions
    n, m = X.shape[0], Y.shape[0]
    a = torch.ones(n, device=device) / n  # Source distribution
    b = torch.ones(m, device=device) / m  # Target distribution

    # Compute optimal transport matrix using Sinkhorn algorithm
    T = sinkhorn_knopp(a, b, C, reg=epsilon, maxIter=max_iter, device=device)

    return T