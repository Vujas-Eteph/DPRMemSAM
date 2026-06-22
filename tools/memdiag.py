"""Memory-information metrics for the DAM4SAM memory bank (numpy only, no heavy deps).

Given a per-frame descriptor matrix D of shape (K, d) — one row per memory frame (init /
DRM / RAM), e.g. the object pointers (d=256) or object-region-pooled mask-memory features
(d=64) — these quantify the *content / diversity / similarity* of the information the memory
holds. All metrics work on L2-normalized rows (direction, not magnitude); no centering, so
K identical frames give effective rank ~1 and novelty ~0.

    cosine_gram(D)        -> (K, K) cosine similarity (redundancy / similarity)
    effective_rank(D)     -> float in [1, K]: entropy-based # of independent directions
    stable_rank(D)        -> float in [1, K]: ||D||_F^2 / sigma_max^2
    per_frame_novelty(D)  -> (K,) residual fraction of each frame vs. the span of the rest
                             (1 = wholly new information, 0 = fully explained by the others)
"""

from __future__ import annotations

import numpy as np


def _l2norm(D: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + eps)


def cosine_gram(D: np.ndarray) -> np.ndarray:
    Dn = _l2norm(D)
    return np.clip(Dn @ Dn.T, -1.0, 1.0).astype(np.float32)


def _singular_values(D: np.ndarray) -> np.ndarray:
    if D.shape[0] == 0:
        return np.zeros(0)
    return np.linalg.svd(_l2norm(D), compute_uv=False)


def effective_rank(D: np.ndarray) -> float:
    """exp(entropy of the normalized singular-value distribution) — a smooth count of the
    independent directions the K frames span. ~1 = all frames alike, up to K = all distinct."""
    s = _singular_values(D)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(D: np.ndarray) -> float:
    """||D||_F^2 / sigma_max^2 — a robust, cheap rank proxy (energy spread across directions)."""
    s = _singular_values(D)
    if s.size == 0 or s[0] <= 1e-12:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2))


def log_volume(D: np.ndarray) -> float:
    """0.5*log det(Gram) = sum of log singular values of the unit-normalized rows: the
    log-volume of the parallelepiped the K (unit) descriptors span. 0 = orthonormal
    (maximally diverse), -> -inf = collinear (redundant). slogdet for stability."""
    Dn = _l2norm(D)
    G = Dn @ Dn.T
    sign, logdet = np.linalg.slogdet(G + 1e-12 * np.eye(G.shape[0]))
    return float(0.5 * logdet) if sign > 0 else float("-inf")


def volume(D: np.ndarray) -> float:
    """sqrt(det(Gram)) of the unit descriptors, in [0, 1]: 1 = orthonormal (fully diverse),
    0 = collinear/redundant. Intuitive 'how much space is spanned', but underflows to ~0 once
    any two frames are near-parallel (use effective_rank for a graded headline)."""
    lv = log_volume(D)
    return float(np.exp(lv)) if np.isfinite(lv) else 0.0


# ----------------------------------------------------------------------
# Subspace (principal-angle) variant — position/permutation-invariant.
# Instead of one vector per frame, each frame is the PCA *subspace* of its object-region
# feature cloud (the dominant feature directions, regardless of where the object sits). A
# frame is summarized by the d x d scatter matrix C = X X^T / n (X = object pixels, d=64);
# pca_basis turns C into the top-r orthonormal directions covering `energy` of the variance.
# Frames are then compared by principal angles between their subspaces (mean cos^2), so a
# moving / rescaled object whose appearance content is unchanged still scores high.
# ----------------------------------------------------------------------
def pca_basis(C: np.ndarray, energy: float = 0.9) -> np.ndarray:
    """Top-r principal directions (d x r, orthonormal) of a PSD scatter/covariance C,
    r = smallest count whose eigenvalues cover `energy` of the total."""
    C = np.asarray(C, dtype=np.float64)
    w, V = np.linalg.eigh(C)            # ascending
    w = np.clip(w[::-1], 0, None)
    V = V[:, ::-1]
    tot = w.sum()
    if tot <= 1e-12:
        return V[:, :1]
    r = int(np.searchsorted(np.cumsum(w) / tot, energy)) + 1
    return V[:, :max(1, min(r, V.shape[1]))]


def bases_from_scatter(scatters, energy: float = 0.9) -> list[np.ndarray]:
    return [pca_basis(C, energy) for C in scatters]


def subspace_similarity(Ui: np.ndarray, Uj: np.ndarray) -> float:
    """Mean cos^2 of the principal angles between span(Ui) and span(Uj), in [0, 1]
    (1 = identical subspace). Handles different dims via min(r_i, r_j)."""
    s = np.linalg.svd(Ui.T @ Uj, compute_uv=False)  # cos of principal angles
    r = min(Ui.shape[1], Uj.shape[1])
    return float((s[:r] ** 2).sum() / max(1, r))


def subspace_gram(bases) -> np.ndarray:
    K = len(bases)
    S = np.eye(K, dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            S[i, j] = S[j, i] = subspace_similarity(bases[i], bases[j])
    return S


def subspace_novelty(bases) -> np.ndarray:
    """For each frame, 1 - (its highest subspace similarity to any other frame): 1 = its
    appearance subspace is unlike every other frame's, 0 = another frame's matches it.

    (We use max-similarity rather than 'residual orthogonal to the union of the others'
    because once a few frames are present their subspaces jointly span most of the ambient
    feature space, so the residual definition saturates to 0 for everyone.)"""
    K = len(bases)
    if K <= 1:
        return np.ones(K, dtype=np.float64)
    S = subspace_gram(bases)
    return np.array([1.0 - max(S[i, j] for j in range(K) if j != i) for i in range(K)])


def _stack(bases) -> np.ndarray:
    return np.concatenate(bases, axis=1) if bases else np.zeros((1, 1))


def subspace_eff_rank(bases) -> float:
    """Independent feature directions across the union of all frame-subspaces (in [1, d])."""
    s = np.linalg.svd(_stack(bases), compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def subspace_stable_rank(bases) -> float:
    s = np.linalg.svd(_stack(bases), compute_uv=False)
    if s.size == 0 or s[0] <= 1e-12:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2))


def per_frame_novelty(D: np.ndarray) -> np.ndarray:
    """For each frame i, the residual fraction of its (unit) descriptor after projecting onto
    the span of all the other frames: 1 = carries information none of the others do, 0 = it's
    a linear combination of the others. Great for asking 'does the DRM frame add anything new?'.
    """
    X = _l2norm(D)
    K = X.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for i in range(K):
        xi = X[i]
        others = np.delete(X, i, axis=0)
        if others.shape[0] == 0:
            out[i] = 1.0  # a lone frame is entirely "novel"
            continue
        coef, *_ = np.linalg.lstsq(others.T, xi, rcond=None)  # others^T c ~= xi
        resid = float(np.linalg.norm(xi - others.T @ coef))   # xi is unit-norm
        out[i] = min(1.0, max(0.0, resid))
    return out


# ----------------------------------------------------------------------
# Diverse-subset selection — "which frames should the memory keep?"
# Both work from a similarity *kernel* `gram` (K x K, ~unit diagonal): cosine_gram for the
# vector descriptors, subspace_gram for the subspace descriptor. Maximizing log det(gram[S, S])
# = maximizing the volume the chosen rows span = picking mutually dissimilar (diverse) frames
# (the MAP of a determinantal point process / D-optimal design).
# ----------------------------------------------------------------------
def greedy_max_volume(gram: np.ndarray, budget: int, keep=()) -> list[int]:
    """Greedily pick `budget` indices maximizing log det(gram[S, S]) — the max-volume / DPP-MAP
    diverse subset. Uses the fast incremental-Cholesky update of Chen et al. (2018), O(budget*K).
    Returns the picked indices in selection order; stops early if no remaining frame adds volume
    (its conditional gain hits ~0, i.e. it is in the span of those already picked).

    `keep` = indices forced into the subset first (e.g. the init frame, pinned because the real
    DAM4SAM bank stores it permanently); the greedy then fills the remaining budget - len(keep)
    slots around them with the most-diverse frames."""
    G = np.asarray(gram, dtype=np.float64)
    K = G.shape[0]
    budget = max(1, min(int(budget), K))
    keep = [int(k) for k in keep if 0 <= int(k) < K][:budget]
    cis = np.zeros((budget, K))             # running Cholesky factors of the selected block
    di2 = np.clip(np.diag(G).copy(), 1e-12, None)  # conditional gain of each item given the picks
    selected: list[int] = []
    for it in range(budget):
        if it < len(keep):
            j = keep[it]                     # forced (pinned) pick
        else:
            j = int(np.argmax(di2))
            if di2[j] <= 1e-12:
                break                        # nothing left adds independent volume
        selected.append(j)
        if len(selected) == budget:
            break
        if di2[j] <= 1e-12:                  # pinned but redundant: mark selected, no update
            di2[selected] = -np.inf
            continue
        ci = (G[j] - cis[:it].T @ cis[:it, j]) / np.sqrt(di2[j])
        cis[it] = ci
        di2 = di2 - ci ** 2
        di2[selected] = -np.inf              # never reselect a chosen frame
    return selected


# ----------------------------------------------------------------------
# Bures-Wasserstein (2-Wasserstein between the per-frame Gaussian fits). Each frame's object-
# region feature cloud is summarized as N(mu, Sigma); the 2-Wasserstein between two Gaussians has
# the closed form ||mu_i - mu_j||^2 + Tr(Si + Sj - 2 (Si^1/2 Sj Si^1/2)^1/2). The first term is the
# centroid L2 distance (LOCATION); the Bures term is the covariance mismatch (SPREAD / orientation)
# -> one metric that fuses "how far apart" and "how differently shaped" the clouds are. This is the
# W2 between the *Gaussian fits*: exact if the clouds are Gaussian, a 2nd-moment approximation
# otherwise (it ignores multimodality / skew). mu = the stored mean-pool `feat`; Sigma = the stored
# uncentered scatter minus mu mu^T (the centered covariance).
# ----------------------------------------------------------------------
def _psd_sqrt(S: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root of a (near-)PSD matrix via eigendecomposition (negatives clipped)."""
    w, V = np.linalg.eigh((np.asarray(S, dtype=np.float64) + np.asarray(S).T) / 2.0)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def bures_wasserstein2(mu_i, S_i, mu_j, S_j) -> float:
    """Squared 2-Wasserstein between N(mu_i, S_i) and N(mu_j, S_j)."""
    d = np.asarray(mu_i, dtype=np.float64) - np.asarray(mu_j, dtype=np.float64)
    Sih = _psd_sqrt(S_i)
    M = Sih @ np.asarray(S_j, dtype=np.float64) @ Sih
    cross = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2.0), 0.0, None)).sum()
    bures = float(np.trace(S_i) + np.trace(S_j) - 2.0 * cross)
    return float(d @ d) + max(0.0, bures)


def bures_wasserstein_matrix(mus, covs) -> np.ndarray:
    """(K, K) 2-Wasserstein DISTANCE matrix (symmetric, zero diagonal) between the per-frame
    Gaussians (mus[k], covs[k]). The matrix square roots are precomputed once per frame."""
    K = len(mus)
    halves = [_psd_sqrt(C) for C in covs]
    traces = [float(np.trace(C)) for C in covs]
    D = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            M = halves[i] @ np.asarray(covs[j], dtype=np.float64) @ halves[i]
            cross = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2.0), 0.0, None)).sum()
            dmu = np.asarray(mus[i], dtype=np.float64) - np.asarray(mus[j], dtype=np.float64)
            w2 = float(dmu @ dmu) + max(0.0, traces[i] + traces[j] - 2.0 * cross)
            D[i, j] = D[j, i] = np.sqrt(max(0.0, w2))
    return D


def bures_wasserstein2_with_half(mu_i, half_i, tr_i, mu_j, cov_j, tr_j) -> float:
    """2-Wasserstein (not squared) between N(mu_i, S_i) and N(mu_j, S_j) when S_i's PSD square root
    `half_i` (= S_i^1/2) and the traces tr_i = Tr(S_i), tr_j = Tr(S_j) are already known. Lets a
    streaming caller add one new frame against a pool in O(K) cross-terms instead of rebuilding the
    whole bures_wasserstein_matrix each step (the full matrix is O(K^2) eigendecompositions)."""
    half_i = np.asarray(half_i, dtype=np.float64)
    M = half_i @ np.asarray(cov_j, dtype=np.float64) @ half_i
    cross = np.sqrt(np.clip(np.linalg.eigvalsh((M + M.T) / 2.0), 0.0, None)).sum()
    dmu = np.asarray(mu_i, dtype=np.float64) - np.asarray(mu_j, dtype=np.float64)
    w2 = float(dmu @ dmu) + max(0.0, float(tr_i) + float(tr_j) - 2.0 * cross)
    return float(np.sqrt(max(0.0, w2)))


def rank_norm_sim(S: np.ndarray) -> np.ndarray:
    """Rank-normalize a symmetric similarity's off-diagonal entries to evenly fill [0, 1] (diagonal
    forced to 1) — so two similarities of different spread contribute comparably when blended."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    out = np.eye(n)
    if n < 2:
        return out
    iu = np.triu_indices(n, k=1)
    v = S[iu]
    r = np.argsort(np.argsort(v)) / max(1, v.size - 1)
    out[iu] = r
    out[(iu[1], iu[0])] = r
    return out


def blend_similarity_from(cos_gram: np.ndarray, w2: np.ndarray, op: str = "product",
                          lam: float = 0.5, ranknorm: bool = True) -> np.ndarray:
    """Combined pointer⊕Wasserstein SIMILARITY (unit diagonal) from a precomputed cosine Gram (in
    [-1,1]) and 2-Wasserstein distance matrix `w2`. Mirrors the dashboard's `ptr_gauss` descriptor:
    map cosine to [0,1] via (1+cos)/2, the Wasserstein to a similarity via the median-bandwidth RBF
    kernel exp(-(w2/median)^2), optionally rank-normalize each, then combine with `op` ('product'
    AND-like / λ-free, or 'sum' = λ·sim_W + (1-λ)·sim_ptr). Takes precomputed matrices so a streaming
    caller can pass an incrementally-cached `w2` (the cosine half is cheap to recompute). Used as the
    similarity Gram for the diversity selection and as 1 - this for the medoid distance."""
    sim_ptr = (np.asarray(cos_gram, dtype=np.float64) + 1.0) / 2.0
    w2 = np.asarray(w2, dtype=np.float64)
    pos = w2[w2 > 0]
    bw = float(np.median(pos)) if pos.size else 1.0
    sim_w = np.exp(-(w2 / (bw + 1e-12)) ** 2)
    np.fill_diagonal(sim_w, 1.0)
    if ranknorm:
        sim_ptr, sim_w = rank_norm_sim(sim_ptr), rank_norm_sim(sim_w)
    g = sim_w * sim_ptr if op == "product" else lam * sim_w + (1.0 - lam) * sim_ptr
    np.fill_diagonal(g, 1.0)
    return g


def blend_distance_from(cos_gram: np.ndarray, w2: np.ndarray, op: str = "product",
                        lam: float = 0.5, ranknorm: bool = True) -> np.ndarray:
    """Combined pointer⊕Wasserstein DISTANCE = 1 - blend_similarity_from(...) (clipped to >= 0)."""
    return np.clip(1.0 - blend_similarity_from(cos_gram, w2, op, lam, ranknorm), 0.0, None)


def kmedoids(dist: np.ndarray, k: int, fixed=(), n_init: int = 8,
             max_iter: int = 100, seed: int = 0):
    """k-medoids (PAM-style Voronoi iteration) on a *precomputed* (n, n) distance matrix — so it
    clusters directly in the metric space (principal-angle distance for the subspace descriptor,
    1 - cosine for the vectors), not a 2D/3D embedding of it. `fixed` = indices forced to remain
    medoids and seeded first (so fixed[0] is cluster 0) — e.g. the init frame anchored as the
    reference for cluster 0. Returns (labels, medoids) minimizing the total point-to-medoid
    distance, the best of `n_init` random restarts. Each medoid is its own cluster member (its
    self-distance is 0), so clusters never go empty."""
    dist = np.asarray(dist, dtype=np.float64)
    n = dist.shape[0]
    k = max(1, min(int(k), n))
    fixed = [int(f) for f in fixed if 0 <= int(f) < n][:k]
    rng = np.random.default_rng(seed)
    restarts = 1 if len(fixed) >= k else max(1, n_init)
    best = None  # (cost, labels, medoids)
    for _ in range(restarts):
        pool = [i for i in range(n) if i not in fixed]
        extra = list(rng.permutation(pool))[: k - len(fixed)]
        medoids = (list(fixed) + extra)[:k]
        for _ in range(max_iter):
            labels = np.argmin(dist[:, medoids], axis=1)
            new = list(medoids)
            for c in range(len(fixed), k):                  # fixed medoids never move
                members = np.where(labels == c)[0]
                if members.size:
                    new[c] = int(members[int(np.argmin(dist[np.ix_(members, members)].sum(axis=1)))])
            if new == medoids:
                break
            medoids = new
        labels = np.argmin(dist[:, medoids], axis=1)
        cost = float(dist[np.arange(n), [medoids[l] for l in labels]].sum())
        if best is None or cost < best[0]:
            best = (cost, labels, list(medoids))
    return best[1], best[2]


def online_diverse_select(gram: np.ndarray, order, budget: int, keep=()) -> list[int]:
    """Causal streaming selection: walk the indices in temporal `order`, keeping at most `budget`
    of them. When adding a frame exceeds the budget, evict the kept frame whose removal leaves the
    largest log det(gram) among the rest (i.e. the least-diverse member). Only the seen-so-far
    prefix is ever used — an evicted frame cannot be recalled — so this mimics a budget-bounded
    online memory policy, to be compared against what the tracker actually stored. Returns the
    final kept indices.

    `keep` = indices that are PINNED (never evicted) and seeded into the bank first — e.g. the init
    frame, which the real DAM4SAM bank stores permanently. Only the remaining budget - len(keep)
    slots churn under the eviction rule, so a pinned init is always present in the result."""
    G = np.asarray(gram, dtype=np.float64)
    budget = max(1, int(budget))
    keepset = {int(k) for k in keep}
    eye = lambda m: 1e-9 * np.eye(m)
    kept: list[int] = list(keepset)           # pinned frames occupy permanent slots
    for t in order:
        t = int(t)
        if t in keepset:
            continue                          # already pinned in the bank
        kept.append(t)
        if len(kept) > budget:
            best_rest, best_logdet = None, -np.inf
            for c in kept:                    # try evicting each member; keep the most-diverse rest
                if c in keepset:
                    continue                  # never evict a pinned frame
                rest = [x for x in kept if x != c]
                sign, logdet = np.linalg.slogdet(G[np.ix_(rest, rest)] + eye(len(rest)))
                ld = float(logdet) if sign > 0 else -np.inf
                if ld > best_logdet:
                    best_logdet, best_rest = ld, rest
            if best_rest is not None:         # the log-det rule found an evictable, positive-def rest
                kept = best_rest
            else:
                # No removal yields a positive-definite rest — happens when `gram` is not PSD (e.g. a
                # rank-normalized blend), so log det(rest) is undefined for every candidate. Fall back
                # to evicting the most-redundant evictable frame (highest total similarity to the
                # others) so the budget is still respected. Only pinned frames left -> nothing to do.
                evictable = [c for c in kept if c not in keepset]
                if evictable:
                    worst = max(evictable,
                                key=lambda c: float(G[c, [x for x in kept if x != c]].sum()))
                    kept = [x for x in kept if x != worst]
    return kept


def online_medoid_select(dist: np.ndarray, order, budget: int, keep=()) -> list[int]:
    """Streaming online k-medoids with a budget of `budget` prototypes (clusters). Maintains a set
    of clusters, each a set of MEMBER frames with a medoid (its most-central member — an actual
    frame). Walking the indices in temporal `order`:
      - every new frame starts its own cluster;
      - whenever the number of clusters exceeds the budget, the two most similar clusters (closest
        medoids) are MERGED and the merged cluster's medoid is recomputed over ALL its members.
    So a cluster's representative is updated as more of its members arrive — a newly arrived frame
    becomes the medoid if it is more central than the previous one, while the previous members are
    kept and still count toward the recomputation. A genuinely novel frame (not part of the closest
    pair) stays its own cluster and earns a prototype slot. `keep` = pinned indices (e.g. the init
    frame): each is a fixed cluster whose medoid never changes and is never merged away. Returns the
    medoid frame of each final cluster.

    Only the *number* of prototypes is bounded by the budget; the member sets retain the seen frames
    so medoids can be recomputed (a simulation convenience — a real bounded tracker would keep
    running per-cluster statistics instead). Greedy + causal, so it can reach a different (generally
    less optimal) clustering than the batch oracle (k-medoids over all frames at once)."""
    D = np.asarray(dist, dtype=np.float64)
    budget = max(1, int(budget))
    keepset = {int(k) for k in keep}

    def _medoid(members):  # the member minimizing total distance to the others (an actual frame)
        if len(members) == 1:
            return members[0]
        sub = D[np.ix_(members, members)]
        return members[int(sub.sum(axis=1).argmin())]

    clusters = [{"members": [k], "medoid": k, "fixed": True} for k in sorted(keepset)]
    for t in order:
        t = int(t)
        if t in keepset:
            continue
        clusters.append({"members": [t], "medoid": t, "fixed": False})
        while len(clusters) > budget:
            free = [i for i, c in enumerate(clusters) if not c["fixed"]]
            if len(free) < 2:                 # cannot merge two free clusters: drop the newest free
                if free:
                    clusters.pop(free[-1])
                else:
                    break                     # only pinned clusters left (budget < #pinned)
                continue
            ia, ib, best = free[0], free[1], float("inf")
            for x in range(len(free)):
                for y in range(x + 1, len(free)):
                    d = D[clusters[free[x]]["medoid"], clusters[free[y]]["medoid"]]
                    if d < best:
                        best, ia, ib = d, free[x], free[y]
            members = clusters[ia]["members"] + clusters[ib]["members"]
            for i in sorted((ia, ib), reverse=True):
                clusters.pop(i)
            clusters.append({"members": members, "medoid": _medoid(members), "fixed": False})
    return [c["medoid"] for c in clusters]
