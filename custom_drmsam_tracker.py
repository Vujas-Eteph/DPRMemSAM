import numpy as np
import torch
import torch.nn.functional as F

from vot.region.shapes import Mask, Rectangle
from vot.region.raster import calculate_overlaps
from sam2.build_sam import build_sam2_video_predictor
from utils.utils import keep_largest_component
from dam4sam_tracker import DAM4SAMTracker

# tools/memdiag.py (numpy-only memory metrics) for the long-term medoid memory. tools/ is a
# sibling of this file with no __init__, so add it to the path explicitly (robust to the caller's cwd).
import sys as _sys
from pathlib import Path as _Path
_TOOLS_DIR = str(_Path(__file__).resolve().parent / "tools")
if _TOOLS_DIR not in _sys.path:
    _sys.path.insert(0, _TOOLS_DIR)
import memdiag


# Default DRM (distractor-resolving memory) add-frame gate thresholds. A frame is
# added to the DRM only when ALL of these hold (see track()). Override any subset
# from dam4sam_config.yaml's `drm:` block (passed in as drm_config) to ablate the
# gate without editing code. Defaults reproduce the paper / baseline behaviour.
DRM_GATE_DEFAULTS = {
    "iou_min": 0.8,         # chosen mask's predicted IoU must exceed this
    "size_ratio_tol": 0.2,  # current area must be within (1 +/- tol) of the median of
                            # recent areas (tol=0.2 -> 0.8..1.2, the old +/-20% band)
    "size_window": 300,     # how many recent frames of area history to keep
    "size_recent": 10,      # use the last N non-empty areas from that history
    "overlap_max": 0.7,     # add to DRM if min bbox IoU(chosen, alternative) <= this
    "min_stride": 5,        # min frames between successive DRM additions
}


class CustomDAM4SAMTracker(DAM4SAMTracker):
    """
    DAM4SAMTracker (default DAM4SAM behaviour) with the VOT-API fix required to run on the
    installed VOT toolkit:

        original (older VOT API):   Mask(arr).convert(RegionType.RECTANGLE)
        installed VOT API:          Rectangle.convert(Mask(arr))

    plus the optional long-term medoid memory (an additive 3rd memory bank). The
    bounding-box and IoU computations use VOT's own authoritative implementation —
    identical math to the baseline, not a numpy approximation. dam4sam_tracker.py is left
    completely untouched.

    Tracker names:
        sam21pp-L / B / S / T   (SAM 2.1)
        sam2pp-L  / B / S / T   (SAM 2)
    """

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __init__(self, tracker_name="sam2pp-T", memory_stride=None, num_maskmem=None,
                 drm_min_stride=5, apply_postprocessing=True, drm_config=None,
                 longterm_config=None):
        super().__init__(tracker_name)

        # SAM2 postprocessing toggle. DAM4SAMTracker builds the predictor with the
        # build default (apply_postprocessing=True -> dynamic_multimask_via_stability,
        # fill_hole_area, mask binarization). The standard SAM2 VOS eval protocol uses
        # False. We rebuild the predictor without it if requested (one extra model load
        # at init). Must run before the memory_stride override below, since rebuilding
        # creates a fresh predictor.
        if not apply_postprocessing:
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint, device="cuda:0",
                apply_postprocessing=False,
            )

        # RAM (recent appearance memory) temporal stride — the "stride between frames
        # saved in memory". The model config yaml sets this to 5; the DAM4SAM paper
        # uses 1 for LVOS. super().__init__ has just built self.predictor from the
        # yaml, so we override the loaded value here. None = keep the config value (5).
        if memory_stride is not None:
            self.predictor.memory_temporal_stride_for_eval = memory_stride

        # RAM number of memory frames (num_maskmem; default 7 = 1 input + 6 past).
        # None keeps the model default. The learned temporal encoding maskmem_tpos_enc
        # has a fixed number of slots (the trained num_maskmem), so a larger value
        # would index out of bounds — we clamp to that and warn.
        if num_maskmem is not None:
            max_slots = self.predictor.maskmem_tpos_enc.shape[0]
            if num_maskmem > max_slots:
                print(f"[CustomDAM4SAMTracker] num_maskmem={num_maskmem} exceeds the "
                      f"trained temporal-encoding size ({max_slots}); clamping to {max_slots}.")
                num_maskmem = max_slots
            self.predictor.num_maskmem = num_maskmem

        # DRM (distractor-resolving memory) settings from the optional dam4sam_config
        # `drm:` block. The add-frame gate thresholds (read by track()) live in self.drm.
        # The DRM frame count `num_frames` is instead the predictor's
        # max_cond_frames_in_attn (how many conditioning frames are attended), so it's
        # applied to the model, not the gate. Start from DRM_GATE_DEFAULTS, fold in the
        # legacy drm_min_stride, then apply the config overrides.
        drm_config = dict(drm_config) if drm_config else {}
        drm_num_frames = drm_config.pop("num_frames", None)
        self.drm = dict(DRM_GATE_DEFAULTS)
        self.drm["min_stride"] = drm_min_stride
        self.drm.update(drm_config)
        self.drm_min_stride = self.drm["min_stride"]  # kept as an alias for compatibility
        if drm_num_frames is not None:
            self.predictor.max_cond_frames_in_attn = drm_num_frames

        # Long-term medoid memory (optional 3rd bank, additive to DRM + RAM). Off by default, so the
        # tracker is byte-identical to the baseline unless enabled via dam4sam_config.yaml's
        # `longterm:` block. When on, track() maintains a causal pool of object pointers from frames
        # passing the IoU + presence gates and selects `size` representatives (online k-medoids /
        # k-medoids on 1-cosine) each frame; the model attends them as extra conditioning memory.
        lt = dict(longterm_config) if longterm_config else {}
        self.lt_enabled = bool(lt.get("enabled", False))
        self.lt_size = int(lt.get("size", 7))
        self.lt_iou_min = float(lt.get("iou_min", 0.0))
        self.lt_presence_min = float(lt.get("presence_min", 0.0))
        self.lt_method = str(lt.get("method", "online_medoid"))   # "online_medoid" | "kmedoids"
        # Descriptor driving the medoid DISTANCE (mirrors the visualizer's sources):
        #   "ptr"   = object-pointer cosine (256-d; cheapest, always available),
        #   "gauss" = masked mask-memory embedding's Gaussian fit, 2-Wasserstein (Bures) distance,
        #   "blend" = pointer ⊕ Wasserstein combined similarity (product / λ-sum).
        self.lt_descriptor = str(lt.get("descriptor", "ptr"))     # "ptr" | "gauss" | "blend"
        self.lt_region = str(lt.get("region", "fg"))              # "fg" (object) | "complete"
        self.lt_blend_op = str(lt.get("blend_op", "product"))     # "product" | "sum"
        self.lt_blend_lambda = float(lt.get("blend_lambda", 0.5))
        self.lt_blend_ranknorm = bool(lt.get("blend_ranknorm", True))
        self.lt_max_pool = int(lt.get("max_pool", 400))           # cap eligible-pool size (cost)
        # Attend the init frame ONLY via the long-term bank (not as a DRM conditioning frame), so the
        # DRM + RAM budget (num_maskmem) is freed of it. Requires the long-term bank enabled (the init
        # is always pinned in it). Off by default → baseline behaviour.
        self.lt_init_only = bool(lt.get("init_only_in_longterm", False)) and self.lt_enabled
        self._lt_need_ptr = self.lt_descriptor in ("ptr", "blend")
        self._lt_need_gauss = self.lt_descriptor in ("gauss", "blend")
        self.predictor._longterm_frames = []                      # read by sam2_base's memory forward
        self.predictor._lt_exclude_init_from_cond = self.lt_init_only  # read by the memory forward
        self._reset_longterm_pool()

    # -------------------------------------------------------------------------
    # Long-term medoid memory (optional 3rd bank) — causal bookkeeping
    # -------------------------------------------------------------------------

    def initialize(self, image, init_mask, bbox=None):
        """Baseline initialize() + reset the long-term medoid memory and seed it with the init
        frame's descriptor (frame 0 is always eligible, never gated)."""
        out_dict = super().initialize(image, init_mask, bbox=bbox)
        self.predictor._longterm_frames = []
        self._reset_longterm_pool()
        if self.lt_enabled:
            self._admit_longterm(0, m_iou=float("nan"), force=True)  # init frame always eligible
        return out_dict

    def _reset_longterm_pool(self):
        """Clear the eligible-frame pool and the cached pairwise Wasserstein distances. Per pool
        frame we keep only the cheap DESCRIPTOR (object pointer and/or the masked-embedding Gaussian
        μ, Σ, Σ^1/2, Tr Σ) — NOT a copy of the memory tokens, which already live in SAM2's
        output_dict and are fetched by frame index at attention time."""
        self._lt_pool_idx = []                                  # eligible frame indices (temporal)
        self._lt_pool_ptr = []                                  # (256,) object pointers (ptr/blend)
        self._lt_pool_mu = []                                   # (64,) mean-pool   (gauss/blend)
        self._lt_pool_cov = []                                  # (64,64) covariance (gauss/blend)
        self._lt_pool_half = []                                 # (64,64) Σ^1/2     (gauss/blend)
        self._lt_pool_tr = []                                   # Tr Σ              (gauss/blend)
        self._lt_w2 = np.zeros((0, 0))                          # cached 2-Wasserstein distances

    def _frame_output(self, frame_idx):
        """The just-stored per-object SAM2 output for `frame_idx` (cond or non-cond), or None."""
        obj_idx = self.predictor._obj_id_to_idx(self.inference_state, 0)
        od = self.inference_state["output_dict_per_obj"][obj_idx]
        return od["cond_frame_outputs"].get(frame_idx,
                                            od["non_cond_frame_outputs"].get(frame_idx, None))

    def _lt_descriptor(self, o):
        """Compute this frame's long-term descriptor pieces from its stored output: the object
        pointer (ptr/blend) and/or the masked mask-memory embedding's Gaussian fit N(μ, Σ) over the
        chosen region (gauss/blend). Σ^1/2 and Tr Σ are precomputed so a new frame can be added to the
        cached Wasserstein matrix in O(pool) cross-terms. Returns (ptr, mu, cov, half, tr); a piece is
        None when not needed or not computable (e.g. no maskmem_features)."""
        ptr = mu = cov = half = tr = None
        if self._lt_need_ptr and o.get("obj_ptr") is not None:
            ptr = o["obj_ptr"].detach().reshape(-1).float().cpu().numpy()
        if self._lt_need_gauss and o.get("maskmem_features") is not None:
            grid = o["maskmem_features"][0].float()            # (C, Hm, Wm)
            C, Hm, Wm = grid.shape
            grid = grid.reshape(C, -1)                          # (C, Hm*Wm)
            sel = None
            if self.lt_region == "fg":                         # object region (pred_masks > 0)
                pm = o.get("pred_masks")
                if pm is not None:
                    mres = F.interpolate(pm.float(), size=(Hm, Wm), mode="bilinear",
                                         align_corners=False)
                    sel = (mres[0, 0].reshape(-1) > 0)
            X = grid[:, sel] if (sel is not None and bool(sel.any())) else grid
            mu = X.mean(dim=1).cpu().numpy().astype(np.float64)
            scat = (X @ X.t() / max(1, X.shape[1])).cpu().numpy().astype(np.float64)
            cov = scat - np.outer(mu, mu)
            half = memdiag._psd_sqrt(cov)
            tr = float(np.trace(cov))
        return ptr, mu, cov, half, tr

    def _lt_drop(self, i):
        """Remove pool entry `i` from every parallel list and from the cached Wasserstein matrix."""
        for arr in (self._lt_pool_idx, self._lt_pool_ptr, self._lt_pool_mu,
                    self._lt_pool_cov, self._lt_pool_half, self._lt_pool_tr):
            arr.pop(i)
        if self._lt_w2.shape[0] > i:
            self._lt_w2 = np.delete(np.delete(self._lt_w2, i, axis=0), i, axis=1)

    def _admit_longterm(self, frame_idx, m_iou, force=False):
        """Gate frame `frame_idx` into the eligible long-term pool by predicted IoU + object presence
        (sigmoid of object_score_logits), cache its descriptor (incrementally extending the pairwise
        Wasserstein matrix for gauss/blend), then recompute the medoid bank. `force` bypasses the
        gates (the always-eligible init frame). A gate value <= 0 is treated as 'off'."""
        if frame_idx in self._lt_pool_idx:
            return
        o = self._frame_output(frame_idx)
        if o is None:
            return
        osl = o.get("object_score_logits")
        presence = (float(torch.sigmoid(torch.as_tensor(osl).float().reshape(-1)[0]))
                    if osl is not None else float("nan"))
        ok_iou = self.lt_iou_min <= 0 or (np.isfinite(m_iou) and m_iou >= self.lt_iou_min)
        ok_pres = self.lt_presence_min <= 0 or (np.isfinite(presence) and presence >= self.lt_presence_min)
        if not (force or (ok_iou and ok_pres)):
            return
        ptr, mu, cov, half, tr = self._lt_descriptor(o)
        if (self._lt_need_ptr and ptr is None) or (self._lt_need_gauss and mu is None):
            return                                             # descriptor not computable → skip
        self._lt_pool_idx.append(int(frame_idx))
        self._lt_pool_ptr.append(ptr)
        self._lt_pool_mu.append(mu); self._lt_pool_cov.append(cov)
        self._lt_pool_half.append(half); self._lt_pool_tr.append(tr)
        if self._lt_need_gauss:                                # extend the cached Wasserstein matrix
            n = len(self._lt_pool_idx)
            w2 = np.zeros((n, n))
            w2[: n - 1, : n - 1] = self._lt_w2
            for j in range(n - 1):
                d = memdiag.bures_wasserstein2_with_half(
                    mu, half, tr, self._lt_pool_mu[j], self._lt_pool_cov[j], self._lt_pool_tr[j])
                w2[n - 1, j] = w2[j, n - 1] = d
            self._lt_w2 = w2
        if len(self._lt_pool_idx) > self.lt_max_pool:          # cap cost; drop oldest non-init
            self._lt_drop(1 if self._lt_pool_idx[0] == 0 else 0)
        self._recompute_longterm_bank()                        # only when the pool changed

    def _recompute_longterm_bank(self):
        """Select `lt_size` representatives from the eligible pool under the configured descriptor and
        method, and publish their frame indices to the model. <= lt_size pool frames are all kept.
        Builds both a SIMILARITY gram (for the diversity objectives) and a DISTANCE (for the medoid
        objectives), mirroring the visualizer; the init frame is pinned in every method. Methods:
          online_medoid  — causal streaming k-medoids on the distance (most REPRESENTATIVE; viz online)
          kmedoids       — batch k-medoids on the distance (representative; viz oracle)
          online_diverse — causal streaming max-volume on the gram (most DIVERSE/spread; viz online)
          max_volume     — batch greedy max-volume / DPP-MAP on the gram (diverse; viz oracle)."""
        idxs = self._lt_pool_idx
        if len(idxs) <= self.lt_size:
            self.predictor._longterm_frames = list(idxs)
            return
        # Per-descriptor similarity gram (unit diagonal) and the matching distance (1 - similarity,
        # except gauss where the distance is the raw 2-Wasserstein and the gram its RBF kernel).
        if self.lt_descriptor == "gauss":
            w2 = self._lt_w2
            pos = w2[w2 > 0]
            bw = float(np.median(pos)) if pos.size else 1.0
            gram = np.exp(-(w2 / (bw + 1e-12)) ** 2)
            np.fill_diagonal(gram, 1.0)
            dist = w2.copy()
        elif self.lt_descriptor == "blend":
            cos = memdiag.cosine_gram(np.vstack(self._lt_pool_ptr)).astype(np.float64)
            gram = memdiag.blend_similarity_from(cos, self._lt_w2, self.lt_blend_op,
                                                 self.lt_blend_lambda, self.lt_blend_ranknorm)
            dist = np.clip(1.0 - gram, 0.0, None)
        else:  # "ptr"
            gram = memdiag.cosine_gram(np.vstack(self._lt_pool_ptr)).astype(np.float64)
            dist = np.clip(1.0 - gram, 0.0, None)
        init_local = idxs.index(0) if 0 in idxs else None
        keep = (init_local,) if init_local is not None else ()
        order = list(range(len(idxs)))                         # temporal (pool) order
        if self.lt_method == "kmedoids":
            sel = list(memdiag.kmedoids(dist, self.lt_size, fixed=keep)[1])
        elif self.lt_method == "max_volume":
            sel = memdiag.greedy_max_volume(gram, self.lt_size, keep=keep)
        elif self.lt_method == "online_diverse":
            sel = memdiag.online_diverse_select(gram, order, self.lt_size, keep=keep)
        else:  # "online_medoid" — causal streaming selection in temporal (pool) order
            sel = memdiag.online_medoid_select(dist, order, self.lt_size, keep=keep)
        self.predictor._longterm_frames = [idxs[p] for p in sel]

    # -------------------------------------------------------------------------
    # track() — default DAM4SAM logic, VOT-API-fixed
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def track(self, image, init=False):
        torch.cuda.empty_cache()

        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        for out in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            return_all_masks=True,
        ):
            if len(out) == 3:
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

            else:
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                alternative_masks, out_all_ious = alternative_masks_ious
                m_idx = np.argmax(out_all_ious)
                m_iou = out_all_ious[m_idx]

                n_pixels = (m == 1).sum()
                self.object_sizes.append(n_pixels)
                stride_ok = (self.frame_index - self.last_added > self.drm["min_stride"]) \
                    or self.last_added == -1

                # DISTRACTOR gate (default DAM4SAM heuristic): chosen IoU + size-consistency +
                # an alternative-mask that overlaps the chosen one little enough to be a distractor.
                alternative_masks = [mask for i, mask in enumerate(alternative_masks) if i != m_idx]
                if len(self.object_sizes) > 1 and n_pixels >= 1:
                    obj_sizes_ratio = n_pixels / np.median([
                        size for size in self.object_sizes[-self.drm["size_window"]:] if size >= 1
                    ][-self.drm["size_recent"]:])
                else:
                    obj_sizes_ratio = -1

                size_tol = self.drm["size_ratio_tol"]
                if m_iou > self.drm["iou_min"] and (1.0 - size_tol) <= obj_sizes_ratio <= (1.0 + size_tol) and n_pixels >= 1 and stride_ok:
                    alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8)
                                     for m_ in alternative_masks]

                    chosen_mask_np = m.copy()
                    chosen_bbox = Rectangle.convert(Mask(chosen_mask_np))

                    alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                    alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                    if len(alternative_masks) > 0:
                        alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                        alternative_bboxes = [Rectangle.convert(Mask(m_)) for m_ in alternative_masks]
                        ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]

                        if np.min(np.array(ious)) <= self.drm["overlap_max"]:
                            self.last_added = self.frame_index
                            self.predictor.add_to_drm(
                                inference_state=self.inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=0,
                            )

            break  # propagate_in_video yields exactly one item for max_frame_num_to_track=0

        out_dict = {'pred_mask': m}
        # Long-term medoid memory: admit this (finalized) frame into the eligible pool by its
        # predicted IoU + presence and recompute the bank, so it is available to the NEXT frame's
        # memory-conditioned forward (strictly causal). m_iou exists only on the 4-output path
        # (a non-init frame with alternative candidates); else there is no IoU, so use NaN.
        if self.lt_enabled and not init:
            self._admit_longterm(self.frame_index,
                                 m_iou=m_iou if len(out) == 4 else float("nan"))
        self.inference_state["images"].pop(self.frame_index)
        return out_dict
