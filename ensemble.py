"""
ensemble.py  -  IoU-based ensemble module for the self-training crater detection pipeline.

Called automatically from Auto_iterative_self_training.ipynb at the end of each loop:

    python ensemble.py --gt GT_LABEL_DIR
                       --nargs DET_DIR_1 DET_DIR_2 DET_DIR_3
                       --pathimg IMAGE_DIR
                       --n_models 3
                       --pathsave OUTPUT_LABEL_DIR
                       --loop LOOP_NUMBER

Algorithm (paper Section 2.3):
  1. Collect detections from all n_models for each image.
  2. Unanimous consensus: a detection is accepted only when every model
     has a matching detection with pairwise IoU > IOU_CONSENSUS (0.6).
     Detections agreed upon by fewer than all models are discarded.
  3. Among the matched group, keep the single box with the highest confidence.
  4. Apply a size-dependent confidence threshold before adding to training set:
       - small craters  (<1.5 km, box width < 0.036 normalised): conf >= 0.2
       - medium / large (>=1.5 km):                               conf >= 0.5
  5. Merge accepted pseudo-labels with the existing ground-truth labels and
     save to pathsave for use in the next self-training iteration.
"""

import os
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (paper Section 2.2 and 2.3)
# ---------------------------------------------------------------------------

# At 416x416 px with ~100 m px-1, 1.5 km ≈ 15 px → 15/416 ≈ 0.036 normalised.
SMALL_CRATER_THRESHOLD = 15 / 416

CONF_SMALL  = 0.2   # confidence threshold for small craters  (<1.5 km)
CONF_MEDIUM = 0.5   # confidence threshold for medium/large   (>=1.5 km)

IOU_CONSENSUS    = 0.6   # IoU threshold for unanimous-consensus matching
IOU_GT_DUPLICATE = 0.3   # IoU threshold for suppressing GT duplicates


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert [x_c, y_c, w, h] (normalised) to [x1, y1, x2, y2]."""
    xc, yc, w, h = box
    return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes in [x_c, y_c, w, h] normalised format."""
    a = xywh_to_xyxy(box_a)
    b = xywh_to_xyxy(box_b)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_detections(label_dir: str, stem: str) -> np.ndarray:
    """
    Load YOLOv5 detections (saved with --save-conf) for one image.

    Expected line format:  class  x_c  y_c  w  h  conf
    Returns ndarray of shape [N, 5]:  [x_c, y_c, w, h, conf].
    """
    path = Path(label_dir) / f"{stem}.txt"
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:           # class x y w h conf
                rows.append([float(p) for p in parts[1:6]])
            elif len(parts) == 5:         # class x y w h  (no conf column)
                rows.append([float(p) for p in parts[1:5]] + [1.0])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def load_gt_labels(label_dir: str, stem: str) -> list:
    """
    Load ground-truth YOLO labels (no confidence column).
    Returns a list of raw text lines.
    """
    path = Path(label_dir) / f"{stem}.txt"
    if not path.exists():
        return []
    with open(path) as f:
        return [line.rstrip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Ensemble logic
# ---------------------------------------------------------------------------

def find_consensus(detections_per_model: list, iou_threshold: float = IOU_CONSENSUS) -> list:
    """
    Unanimous-consensus ensemble (paper Section 2.3).

    A candidate detection is accepted only if every model contributes a
    matching detection (pairwise IoU > iou_threshold).  All pairwise IoUs
    are evaluated simultaneously so the result is order-independent.
    Among the matched group the box with the highest confidence is kept.

    Parameters
    ----------
    detections_per_model : list of ndarray, each shape [N_m, 5] (x,y,w,h,conf)
    iou_threshold        : consensus IoU threshold (default 0.6)

    Returns
    -------
    list of ndarray, each shape [5] (x,y,w,h,conf)
    """
    n = len(detections_per_model)
    if n == 0:
        return []
    if n == 1:
        return list(detections_per_model[0])

    accepted = []
    used = [set() for _ in range(n)]

    for i, det0 in enumerate(detections_per_model[0]):
        if i in used[0]:
            continue

        group = [(0, i, det0)]
        valid = True

        for m in range(1, n):
            best_iou, best_j, best_det = 0.0, -1, None
            for j, detm in enumerate(detections_per_model[m]):
                if j in used[m]:
                    continue
                iou = compute_iou(det0[:4], detm[:4])
                if iou > iou_threshold and iou > best_iou:
                    best_iou, best_j, best_det = iou, j, detm
            if best_j == -1:
                valid = False
                break
            group.append((m, best_j, best_det))

        if valid:
            used[0].add(i)
            for m, j, _ in group[1:]:
                used[m].add(j)
            best_det = max(group, key=lambda t: t[2][4])[2]
            accepted.append(best_det)

    return accepted


def apply_confidence_filter(detections: list,
                             small_thr:   float = SMALL_CRATER_THRESHOLD,
                             conf_small:  float = CONF_SMALL,
                             conf_medium: float = CONF_MEDIUM) -> list:
    """
    Size-dependent confidence filter (paper Section 2.2).

    Small craters (box width < small_thr) require conf >= conf_small.
    Medium / large craters require conf >= conf_medium.
    """
    return [
        det for det in detections
        if det[4] >= (conf_small if det[2] < small_thr else conf_medium)
    ]


def is_duplicate_of_gt(det: np.ndarray, gt_lines: list,
                        iou_threshold: float = IOU_GT_DUPLICATE) -> bool:
    """Return True if det overlaps an existing GT label above iou_threshold."""
    for line in gt_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        gt_box = np.array([float(p) for p in parts[1:5]])
        if compute_iou(det[:4], gt_box) > iou_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ensemble(args):
    os.makedirs(args.pathsave, exist_ok=True)

    detection_dirs = args.nargs
    first_dir = Path(detection_dirs[0])
    stems = [p.stem for p in first_dir.glob("*.txt")]

    n_new = 0

    for stem in stems:
        dets_per_model = [load_detections(d, stem) for d in detection_dirs[: args.n_models]]

        consensus = find_consensus(dets_per_model, IOU_CONSENSUS)
        filtered  = apply_confidence_filter(consensus)

        gt_lines  = load_gt_labels(args.gt, stem)
        new_lines = list(gt_lines)

        for det in filtered:
            if not is_duplicate_of_gt(det, gt_lines):
                x, y, w, h = det[:4]
                new_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                n_new += 1

        out_path = Path(args.pathsave) / f"{stem}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"[Loop {args.loop}] Processed {len(stems)} images. "
          f"New pseudo-labels added: {n_new}")


def parse_args():
    p = argparse.ArgumentParser(
        description="IoU-based ensemble pseudo-label selector for crater self-training"
    )
    p.add_argument("--gt",       required=True,
                   help="Directory containing ground-truth YOLO label files")
    p.add_argument("--nargs",    nargs="+", required=True,
                   help="Detection label directories, one per model")
    p.add_argument("--pathimg",  required=True,
                   help="Training image directory (used for reference only)")
    p.add_argument("--n_models", type=int, default=3,
                   help="Number of ensemble models (default: 3)")
    p.add_argument("--pathsave", required=True,
                   help="Output directory for merged pseudo-labels")
    p.add_argument("--loop",     type=int, default=1,
                   help="Current self-training loop index (for logging)")
    return p.parse_args()


if __name__ == "__main__":
    run_ensemble(parse_args())
