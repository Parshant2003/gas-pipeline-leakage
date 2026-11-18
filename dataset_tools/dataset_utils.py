"""Shared helpers: source-group detection and image fingerprints.

A "source group" is the set of images that come from the same original photograph.
Roboflow created 3 augmented copies of every source photo *before* the split, and the
dataset also contains the same stock photo at different resolutions, so a plain
filename match is not enough. Images are linked into one group when ANY rule fires:

  R1  same Roboflow source name  (name before "_jpg.rf.<hash>")
  R2  same stock-photo id        (a run of 8+ digits in the file name)
  R3  byte-identical file        (md5)
  R4  dHash Hamming distance <= DHASH_T   (64-bit gradient hash)
  R5  Pearson correlation of 16x16 grayscale thumbnails >= CORR_T
  R6  SIFT keypoints + RANSAC affine agree (>= SIFT_MIN_INLIERS, spread >= SIFT_MIN_SPREAD)
"""
import hashlib
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image

try:  # OpenCV's own thread pool is unsafe after fork(); we parallelise with processes instead
    import cv2
    cv2.setNumThreads(1)
except ImportError:
    pass

DHASH_T = 6
CORR_T = 0.95
SPLITS = ("train", "valid", "test")


def source_name(filename: str) -> str:
    """Strip Roboflow's '_jpg.rf.<hash>.jpg' suffix to get the source photo name."""
    return re.sub(r"_(jpg|jpeg|png|webp)\.rf\..*$", "", filename, flags=re.I)


def stock_id(name: str):
    m = re.search(r"(\d{8,})", name)
    return m.group(1) if m else None


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dhash(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def thumb(img: Image.Image) -> np.ndarray:
    """Zero-mean, unit-norm 16x16 grayscale vector (dot product == Pearson corr)."""
    a = np.asarray(img.convert("L").resize((16, 16), Image.LANCZOS), dtype=np.float64).flatten()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def phash(img: Image.Image) -> np.ndarray:
    """Independent verification hash: 8x8 low-frequency DCT of a 32x32 thumbnail."""
    from scipy.fft import dct
    a = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
    return d[1:] > np.median(d[1:])


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def list_images(root: str):
    """Yield (split, filename, image_path, label_path) for a <root>/<split>/{images,labels} tree."""
    for sp in SPLITS:
        idir = os.path.join(root, sp, "images")
        for f in sorted(os.listdir(idir)):
            lp = os.path.join(root, sp, "labels", os.path.splitext(f)[0] + ".txt")
            yield sp, f, os.path.join(idir, f), lp


def build_groups(records, src_links=()):
    """records: list of dict. src_links: [(src_name_a, src_name_b)] found by SIFT matching.
    Returns (group_id list, merge-reason counts)."""
    n = len(records)
    uf = UnionFind(n)
    why = defaultdict(int)
    # R1 / R2 / R3 : exact keys
    for key in ("src", "sid", "md5"):
        seen = {}
        for i, r in enumerate(records):
            k = r[key]
            if k is None:
                continue
            if k in seen and uf.find(seen[k]) != uf.find(i):
                uf.union(seen[k], i)
                why[key] += 1
            seen.setdefault(k, i)
    # R6 : SIFT + RANSAC links between source photos (crop / stretch / watermark / overlay)
    first = {}
    for i, r in enumerate(records):
        first.setdefault(r["src"], i)
    for a, b in src_links:
        if uf.find(first[a]) != uf.find(first[b]):
            uf.union(first[a], first[b])
            why["sift"] += 1
    # R4 / R5 : perceptual
    H = np.array([r["dh"] for r in records])
    T = np.array([r["th"] for r in records])
    C = T @ T.T
    for i in range(n):
        d = (H != H[i]).sum(1)
        hit = np.where((d <= DHASH_T) | (C[i] >= CORR_T))[0]
        for j in hit:
            if j > i and uf.find(i) != uf.find(j):
                uf.union(i, j)
                why["perceptual"] += 1
    roots = {}
    gid = []
    for i in range(n):
        gid.append(roots.setdefault(uf.find(i), len(roots)))
    return gid, dict(why)


# ----------------------------------------------------------------------------- SIFT matching
# Hash/correlation tests miss the same photo when it is cropped, stretched, has a watermark bar
# or an overlay. Local-feature matching + RANSAC finds those: two images are "the same photo"
# when many SIFT keypoints agree on ONE geometric (affine) transform, spread over the image.
SIFT_MIN_INLIERS = 8
SIFT_MIN_SPREAD = 0.10     # inlier keypoints must cover >= 10% of the image area


def sift_features(path):
    import cv2
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (320, 320), interpolation=cv2.INTER_AREA)
    # ignore the bottom 14%: stock-site watermark bars ("shutterstock.com", "alamy") live there and
    # would make unrelated photos look alike
    mask = np.zeros_like(g); mask[: int(320 * 0.86)] = 255
    kp, ds = cv2.SIFT_create(nfeatures=500).detectAndCompute(g, mask)
    pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
    return pts, ds


def sift_match(f1, f2):
    """Symmetric: the Lowe ratio test is directional, so try both directions, keep the better."""
    return max(_sift_match_one_way(f1, f2), _sift_match_one_way(f2, f1))


def _sift_match_one_way(f1, f2):
    """Return (n_inliers, spread) of the best affine transform from f1 to f2."""
    import cv2
    (p1, d1), (p2, d2) = f1, f2
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return 0, 0.0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return 0, 0.0
    s = p1[[g.queryIdx for g in good]]; d = p2[[g.trainIdx for g in good]]
    M, mask = cv2.estimateAffine2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return 0, 0.0
    # a real re-crop / re-scale / re-stretch of the same photo gives a plausible transform;
    # accidental matches between different photos give degenerate ones -> reject those
    sx, sy = float(np.hypot(M[0, 0], M[1, 0])), float(np.hypot(M[0, 1], M[1, 1]))
    if not (0.4 < sx < 2.5 and 0.4 < sy < 2.5 and 0.5 < sx / sy < 2.0):
        return 0, 0.0
    if abs((M[0, 0] * M[0, 1] + M[1, 0] * M[1, 1]) / (sx * sy)) > 0.35 or np.linalg.det(M[:, :2]) <= 0:
        return 0, 0.0
    mask = mask.ravel().astype(bool)
    n = int(mask.sum())
    if n < 4:
        return 0, 0.0
    q = s[mask]
    spread = float((q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min()) / (320 * 320))
    return n, spread


_F = {}


def is_link(cands, min_in=SIFT_MIN_INLIERS, min_spread=SIFT_MIN_SPREAD, big_in=25, big_spread=0.03):
    """cands: [(inliers, spread), ...]. Linked when many inliers cover a good part of the image, OR
    when there are very many inliers even if they cover only a cropped/zoomed region."""
    return any((n >= min_in and sp >= min_spread) or (n >= big_in and sp >= big_spread) for n, sp in cands)


def _row(a):
    """All candidate matches between key index `a` and every later key (over all their images)."""
    keys = list(_F)
    out = {}
    for b in range(a + 1, len(keys)):
        c = []
        for f1 in _F[keys[a]]:
            for f2 in _F[keys[b]]:
                n, sp = sift_match(f1, f2)
                if n >= 8 and sp >= 0.03:
                    c.append((n, sp))
        out[(keys[a], keys[b])] = c
    return out


def sift_pair_table(paths_by_key, max_imgs=None):
    """paths_by_key: {key: [image_path, ...]}. Up to `max_imgs` images per key are used, because
    one augmented copy (blur / darker / rotated) can match worse than another. Returns
    {(k1, k2): [(inliers, spread), ...]} (candidates only); parallelised over CPU cores."""
    from multiprocessing import Pool
    _F.clear()
    for k, ps in paths_by_key.items():
        _F[k] = [sift_features(p) for p in ps[:max_imgs]]
    n = len(_F)
    out = {}
    with Pool() as pool:
        for d in pool.imap_unordered(_row, range(n), chunksize=4):
            out.update(d)
    return out
