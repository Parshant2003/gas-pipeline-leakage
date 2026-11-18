"""Phase 3a - rebuild the dataset as a leakage-free, source-grouped 70/15/15 split.

Usage:
    python scripts/prepare_dataset.py --src data/original_backup --dst data/gas_grouped

The original folder is only READ. Everything is written to --dst.
No augmentation is performed here (augmentation happens later, on train only).
"""
import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from dataset_utils import (SIFT_MIN_INLIERS, SIFT_MIN_SPREAD, SPLITS, build_groups, dhash,
                           is_link, list_images, md5, sift_pair_table, source_name, stock_id, thumb)

RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
SEED = 42
N_TRIALS = 5000
# A polygon is converted to a box only if it really encloses an object:
POLY_MIN_FILL = 0.30      # polygon area / its bounding-box area
POLY_MIN_AREA = 0.01      # converted box must cover >=1% of the image (smallest real 'Gas' box covers 4.3%)


# ----------------------------------------------------------------------------- labels
def poly_area(pts):
    """Shoelace area of a polygon given as [(x, y), ...]."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def parse_label(path):
    """Return (yolo_box_lines, polygon_report_list). Polygons are verified, not blindly dropped."""
    boxes, polys = [], []
    if not os.path.exists(path):
        return boxes, polys
    for raw in open(path):
        t = raw.split()
        if not t:
            continue
        cls, v = int(float(t[0])), list(map(float, t[1:]))
        if len(t) == 5:
            boxes.append((cls, *v))
            continue
        pts = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        fill = poly_area(pts) / (w * h) if w * h > 0 else 0.0
        ok = fill >= POLY_MIN_FILL and w * h >= POLY_MIN_AREA
        rep = dict(n_points=len(pts), box_w=round(w, 4), box_h=round(h, 4), box_area=round(w * h, 4),
                   fill_ratio=round(fill, 4), converted=ok)
        if ok:
            boxes.append((cls, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, w, h))
        polys.append(rep)
    return boxes, polys


# ----------------------------------------------------------------------------- split
def group_stats(groups, recs):
    st = {}
    for g, idx in groups.items():
        areas = [b[3] * b[4] for i in idx for b in recs[i]["boxes"]]
        st[g] = dict(n=len(idx), boxes=len(areas), area=float(np.sum(areas)))
    return st


def try_split(group_ids, st, total, rng, forced):
    """Randomised greedy: biggest groups first, each goes to the split with the largest deficit."""
    order = sorted(group_ids, key=lambda g: (-st[g]["n"], rng.random()))
    cur = {s: 0 for s in SPLITS}
    assign = {}
    for g, s in forced.items():
        assign[g] = s
        cur[s] += st[g]["n"]
    for g in order:
        if g in assign:
            continue
        n = st[g]["n"]
        best = max(SPLITS, key=lambda s: (RATIOS[s] * total - cur[s]) - 0.0 * n + rng.random() * 0.5)
        assign[g] = best
        cur[best] += n
    return assign, cur


def score(assign, st, total):
    cnt = {s: 0 for s in SPLITS}
    box = {s: 0 for s in SPLITS}
    area = {s: 0.0 for s in SPLITS}
    ng = {s: 0 for s in SPLITS}
    for g, s in assign.items():
        cnt[s] += st[g]["n"]; box[s] += st[g]["boxes"]; area[s] += st[g]["area"]; ng[s] += 1
    e = sum(abs(cnt[s] / total - RATIOS[s]) for s in SPLITS) * 100          # image proportions
    mean_area = {s: area[s] / max(box[s], 1) for s in SPLITS}
    e += sum(abs(mean_area[s] - mean_area["train"]) for s in ("valid", "test")) * 50  # box-size balance
    e += sum(abs(ng[s] / sum(ng.values()) - RATIOS[s]) for s in SPLITS) * 30        # group proportions
    return e


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    # 1. read everything ------------------------------------------------------
    recs = []
    for sp, f, ip, lp in list_images(a.src):
        im = Image.open(ip)
        boxes, polys = parse_label(lp)
        recs.append(dict(orig_split=sp, file=f, img=ip, lbl=lp, src=source_name(f),
                         sid=stock_id(source_name(f)), md5=md5(ip), dh=dhash(im), th=thumb(im),
                         boxes=boxes, polys=polys))
    n = len(recs)
    print(f"read {n} images")

    # 2. polygon verification report ----------------------------------------------
    poly_rows = [dict(file=r["file"], orig_split=r["orig_split"], **p) for r in recs for p in r["polys"]]
    print(f"polygon lines: {len(poly_rows)} | converted: {sum(p['converted'] for p in poly_rows)}")

    # 3. source groups --------------------------------------------------------------
    rep = defaultdict(list)
    for r in recs:
        rep[r["src"]].append(r["img"])
    print(f"SIFT matching {len(rep)} source photos pairwise ...")
    table = sift_pair_table(rep)
    links = [k for k, c in table.items() if is_link(c)]
    gid, why = build_groups(recs, links)
    groups = defaultdict(list)
    for i, g in enumerate(gid):
        groups[g].append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"source groups: {len(groups)} | merge reasons: {why} | largest sizes: {sizes[:8]}")

    # images that lost all their labels (invalid polygons only) are kept as background images
    # and pinned to TRAIN so that the evaluation splits only contain gas-positive images.
    forced = {}
    for g, idx in groups.items():
        if any(not recs[i]["boxes"] for i in idx):
            forced[g] = "train"

    # 4. choose the split -------------------------------------------------------------
    st = group_stats(groups, recs)
    best = None
    for t in range(N_TRIALS):
        rng = random.Random(SEED + t)
        assign, _ = try_split(list(groups), st, n, rng, forced)
        sc = score(assign, st, n)
        if best is None or sc < best[0]:
            best = (sc, assign, SEED + t)
    sc, assign, seed = best
    print(f"best split score {sc:.3f} (seed {seed})")

    # 5. write output ---------------------------------------------------------------------
    if os.path.exists(a.dst):
        shutil.rmtree(a.dst)
    for sp in SPLITS:
        os.makedirs(f"{a.dst}/{sp}/images"); os.makedirs(f"{a.dst}/{sp}/labels")
    manifest = []
    for g, idx in groups.items():
        sp = assign[g]
        for i in idx:
            r = recs[i]
            shutil.copy2(r["img"], f"{a.dst}/{sp}/images/{r['file']}")
            stem = os.path.splitext(r["file"])[0]
            with open(f"{a.dst}/{sp}/labels/{stem}.txt", "w") as fh:
                for c, x, y, w, h in r["boxes"]:
                    fh.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            manifest.append(dict(file=r["file"], group_id=f"G{g:04d}", orig_split=r["orig_split"],
                                 new_split=sp, n_boxes=len(r["boxes"]), polygon_lines=len(r["polys"])))
    manifest.sort(key=lambda m: (m["new_split"], m["group_id"], m["file"]))
    with open(f"{a.dst}/split_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    with open(f"{a.dst}/polygon_verification.json", "w") as fh:
        json.dump(poly_rows, fh, indent=1)

    # data.yaml : 'path: .' makes it location independent (works on Windows 11 too)
    with open(f"{a.dst}/data.yaml", "w") as fh:
        fh.write("# Gas leak detection - source-grouped split (no source-image leakage)\n"
                 "path: .            # folder that contains this file\n"
                 "train: train/images\nval: valid/images\ntest: test/images\n\n"
                 "nc: 1\nnames: ['Gas']\n\n"
                 "roboflow:\n  workspace: video-analytics-nyp1g\n  project: gas-q9ewl\n  version: 3\n"
                 "  license: CC BY 4.0\n  url: https://universe.roboflow.com/video-analytics-nyp1g/gas-q9ewl/dataset/3\n")
    json.dump(dict(seed=seed, ratios=RATIOS, rules=dict(dhash_hamming_max=6, thumb_corr_min=0.95, sift_min_inliers=SIFT_MIN_INLIERS, sift_min_spread=SIFT_MIN_SPREAD),
                   merge_reasons=why), open(f"{a.dst}/split_config.json", "w"), indent=1)
    print("done ->", a.dst)


if __name__ == "__main__":
    main()
