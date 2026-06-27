"""Reference-PCA build + shrinkage-corrected projection (vendored from FRAPOSA).

The functions ``standardize``, ``svd_eigcov``, ``svd_online``, ``procrustes``,
``procrustes_diffdim``, ``ref_aug_procrustes`` and ``oadp`` are adapted (numpy-only,
array-first, logging/file-IO removed) from FRAPOSA / fraposa_pgsc:

    https://github.com/PGScatalog/fraposa_pgsc  (MIT License)
    Copyright (c) Daiwei Zhang (2023) and the PGS Catalog team.

The full upstream MIT notice is vendored alongside this module as ``FRAPOSA_LICENSE``.

Why vendor instead of depend: ``fraposa-pgsc`` pins ``numpy<2.0`` / ``pandas<2.0``,
which conflicts with this workspace's numpy 2.x. The license is clean (MIT); only the
pins are the blocker, so the ~150 lines of projection math are ported here.

OADP (Online Augmentation, Decomposition and Procrustes) corrects the shrinkage bias
that plagues naive PCA projection when #variants >> #samples — the reason FRAPOSA
exists. The runtime is pure numpy (no compiled extension), so it stays Windows-clean.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "standardize",
    "svd_eigcov",
    "svd_online",
    "procrustes",
    "procrustes_diffdim",
    "ref_aug_procrustes",
    "oadp",
    "build_reference_pca",
    "ReferencePCA",
    "project_samples",
    "DEFAULT_MISSING",
]

# Missing-genotype sentinel used in standardisation. read_pgen_genotypes emits -9 for
# missing; we standardise with this value treated as missing (set to the column mean,
# i.e. 0 after centering).
DEFAULT_MISSING: float = -9.0


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    miss: float = DEFAULT_MISSING,
) -> tuple[np.ndarray, np.ndarray]:
    """In-place per-variant standardisation of a (variants x samples) float matrix.

    Missing entries (== ``miss``) are excluded from the mean/std and set to 0 after
    centering. When ``mean``/``std`` are given they are reused (study-sample path);
    otherwise they are computed (reference-build path). Returns ``(mean, std)`` as
    column vectors.
    """
    assert np.issubdtype(X.dtype, np.floating)
    p, _ = X.shape
    is_miss = X == miss
    if (mean is None) or (std is None):
        mean = np.zeros(p)
        std = np.zeros(p)
        for i in range(p):
            row_nomiss = X[i, :][~is_miss[i, :]]
            if row_nomiss.size:
                mean[i] = np.mean(row_nomiss)
                std[i] = np.std(row_nomiss)
        std[std == 0] = 1
    mean = mean.reshape((-1, 1))
    std = std.reshape((-1, 1))
    X -= mean
    X /= std
    X[is_miss] = 0
    return mean, std


def svd_eigcov(XTX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD of a covariance matrix via eigendecomposition. Returns ``(s, V)`` desc."""
    ssq, V = np.linalg.eigh(XTX)
    V = np.squeeze(V)
    ssq = np.squeeze(ssq)
    s = np.sqrt(abs(ssq))
    V = V.T[::-1].T
    s = s[::-1]
    return s, V


def svd_online(
    U1: np.ndarray, d1: np.ndarray, V1: np.ndarray, b: np.ndarray, l: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 online SVD update augmenting the reference SVD with one new sample ``b``."""
    n, k = V1.shape
    if l is None:
        l = k
    assert U1.shape[1] == k
    assert len(d1) == k
    assert l <= k
    p = U1.shape[0]
    b = b.reshape((p, 1))
    b_tilde = b - U1 @ (U1.T @ b)
    b_tilde = b_tilde / np.sqrt(sum(np.square(b_tilde)))
    R = np.concatenate((np.diag(d1), U1.transpose() @ b), axis=1)
    R_tail = np.concatenate((np.zeros((1, k)), b_tilde.transpose() @ b), axis=1)
    R = np.concatenate((R, R_tail), axis=0)
    d2, R_Vt = np.linalg.svd(R, full_matrices=False)[1:]
    V_new = np.zeros((k + 1, n + 1))
    V_new[:k, :n] = V1.transpose()
    V_new[k, n] = 1
    V2 = (R_Vt @ V_new).transpose()[:, :l]
    return d2, V2


def procrustes(
    Y_mat: np.ndarray, X_mat: np.ndarray, return_transformed: bool = False
):
    """Best similarity transform (rotation R, scale rho, translation c) from X to Y."""
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    X_mean = np.mean(X, 0)
    Y_mean = np.mean(Y, 0)
    X -= X_mean
    Y -= Y_mean
    C = Y.T @ X
    U, s, VT = np.linalg.svd(C, full_matrices=False)
    trXX = np.sum(X**2)
    trS = np.sum(s)
    R = VT.T @ U.T
    rho = trS / trXX
    c = Y_mean - rho * X_mean @ R
    if return_transformed:
        X_new = X_mat @ R * rho + c
        return R, rho, c, X_new
    return R, rho, c


def procrustes_diffdim(
    Y_mat: np.ndarray,
    X_mat: np.ndarray,
    n_iter_max: int = 10000,
    epsilon_min: float = 1e-6,
    return_transformed: bool = False,
):
    """Procrustes when X has more dimensions than Y (iterative padding)."""
    X = np.array(X_mat, dtype=np.double, copy=True)
    Y = np.array(Y_mat, dtype=np.double, copy=True)
    n_X, p_X = X.shape
    n_Y, p_Y = Y.shape
    assert n_X == n_Y
    assert p_X >= p_Y
    if p_X == p_Y:
        return procrustes(Y, X, return_transformed)
    Z = np.zeros((n_X, p_X - p_Y))
    R = rho = c = X_new = None
    for _ in range(n_iter_max):
        W = np.hstack((Y, Z))
        R, rho, c = procrustes(W, X)
        X_new = X @ R * rho + c
        Z_new = X_new[:, p_Y:]
        Z_new_centered = Z_new - np.mean(Z_new, 0)
        Z_diff = Z_new - Z
        epsilon = np.sum(Z_diff**2) / np.sum(Z_new_centered**2)
        if epsilon < epsilon_min:
            break
        Z = Z_new
    if return_transformed:
        return R, rho, c, X_new
    return R, rho, c


def ref_aug_procrustes(pcs_ref: np.ndarray, pcs_aug: np.ndarray) -> np.ndarray:
    """Map the augmented (ref + 1 study) PCs back onto the reference PC frame."""
    n_ref, _ = pcs_ref.shape
    n_aug, _ = pcs_aug.shape
    assert n_aug == n_ref + 1
    pcs_aug_head = pcs_aug[:-1, :]
    pcs_aug_tail = pcs_aug[-1, :].reshape((1, -1))
    R, rho, c = procrustes_diffdim(pcs_ref, pcs_aug_head)
    pcs_aug_tail_trsfed = pcs_aug_tail @ R * rho + c
    return pcs_aug_tail_trsfed.flatten()


def oadp(
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    dim_ref: int = 4,
    dim_stu: int | None = None,
    dim_online: int | None = None,
) -> np.ndarray:
    """Project a single standardised study sample ``b`` with shrinkage correction."""
    if dim_stu is None:
        dim_stu = dim_ref * 2
    if dim_online is None:
        dim_online = dim_stu * 2
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    s_aug, V_aug = svd_online(U[:, :dim_online], s[:dim_online], V[:, :dim_online], b)
    s_aug, V_aug = s_aug[:dim_stu], V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    pcs_stu = ref_aug_procrustes(pcs_ref, pcs_aug)
    return pcs_stu[:dim_ref]


# ---------------------------------------------------------------------------
# just-prs wrappers (not from FRAPOSA): reference build + batched projection.
# ---------------------------------------------------------------------------


class ReferencePCA:
    """Reference PCA model produced by :func:`build_reference_pca`.

    Holds everything OADP projection + KNN classification need:
    ``mean``/``std`` (per variant), ``s`` (singular values), ``V`` (n_samples x
    dim_online sample eigenvectors), ``U`` (n_variants x dim_online loadings), and the
    derived ``pcs_ref`` (n_samples x dim_ref reference PC scores).
    """

    __slots__ = ("mean", "std", "s", "V", "U", "dim_ref", "dim_stu", "dim_online")

    def __init__(self, mean, std, s, V, U, dim_ref, dim_stu, dim_online):
        self.mean = mean
        self.std = std
        self.s = s
        self.V = V
        self.U = U
        self.dim_ref = dim_ref
        self.dim_stu = dim_stu
        self.dim_online = dim_online

    @property
    def pcs_ref(self) -> np.ndarray:
        """Reference-sample PC scores (n_samples x dim_ref)."""
        return self.V[:, : self.dim_ref] * self.s[: self.dim_ref]


def build_reference_pca(
    X: np.ndarray, dim_ref: int = 10, missing: float = DEFAULT_MISSING
) -> ReferencePCA:
    """Build a reference PCA from a (variants x samples) genotype dosage matrix.

    Mirrors FRAPOSA's ``pca`` reference-build path: standardise per variant, eigen-
    decompose the sample covariance ``X.T @ X``, then form loadings ``U = X (V/s)``.
    ``dim_stu = 2*dim_ref``, ``dim_online = 2*dim_stu`` (FRAPOSA defaults).
    """
    dim_stu = dim_ref * 2
    dim_online = dim_stu * 2
    Xf = X.astype(np.float64, copy=True)
    mean, std = standardize(Xf, miss=missing)
    n_samples = Xf.shape[1]
    dim_online = min(dim_online, n_samples)
    XTX = Xf.T @ Xf
    s, V = svd_eigcov(XTX)
    V = V[:, :dim_online]
    s = s[:dim_online]
    # Guard against zero singular values when forming loadings.
    s_safe = np.where(s == 0, 1.0, s)
    U = Xf @ (V / s_safe)
    return ReferencePCA(mean, std, s, V, U, dim_ref, dim_stu, dim_online)


def project_samples(model: ReferencePCA, W: np.ndarray, missing: float = DEFAULT_MISSING) -> np.ndarray:
    """OADP-project standardised study genotypes ``W`` (variants x samples) -> PC scores.

    Returns an (n_samples x dim_ref) array. ``W`` is standardised in place against the
    reference ``mean``/``std`` (missing -> column mean) before projection.
    """
    p_stu, n_stu = W.shape
    pcs = np.zeros((n_stu, model.dim_ref))
    for i in range(n_stu):
        w = W[:, i].astype(np.float64).reshape((-1, 1))
        standardize(w, model.mean, model.std, miss=missing)
        pcs[i, :] = oadp(
            model.U, model.s, model.V, w,
            dim_ref=model.dim_ref, dim_stu=model.dim_stu, dim_online=model.dim_online,
        )
    return pcs
