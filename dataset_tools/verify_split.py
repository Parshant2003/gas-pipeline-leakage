"""Phase 3b - verify the grouped split and write data/gas_grouped/dataset_report.md.

Usage:  python scripts/verify_split.py --orig data/original_backup --new data/gas_grouped
Exit code is 1 if ANY check fails.
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from dataset_utils import SPLITS, is_link, sift_pair_table, list_images, md5, phash, source_name, stock_id, thumb, dhash

# Pairs flagged by the loose SIFT rule that were inspected by eye and are DIFFERENT photos
# (the match comes from the identical stock-site watermark box, not from the scene).
REVIEWED_FALSE_POSITIVES = [
    ("istockphoto-1182216719", "istockphoto-866354912"),   # both carry an identical 'iStock / Credit:' box
]
LOOSE_PHASH = 8     # verification thresholds are LOOSER than the grouping thresholds
LOOSE_DHASH = 8
LOOSE_CORR = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--checksums", default=None)
    a = ap.parse_args()
    ok_all, lines = True, []

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        lines.append(f"| {name} | {'PASS' if cond else '**FAIL**'} | {detail} |")
        print(("PASS " if cond else "FAIL ") + name, detail)

    # ---- 1. original untouched ------------------------------------------------------
    a.orig, a.new = os.path.abspath(a.orig), os.path.abspath(a.new)
    chk = a.checksums or os.path.join(os.path.dirname(a.orig), "original_checksums.sha256")
    r = subprocess.run(["sha256sum", "-c", "--quiet", chk], cwd=a.orig, capture_output=True, text=True)
    check("Original backup unchanged (sha256 of all files)", r.returncode == 0, "all checksums match" if r.returncode == 0 else r.stdout[:200])

    # ---- 2. load both trees ------------------------------------------------------------
    man = {m["file"]: m for m in csv.DictReader(open(os.path.join(a.new, "split_manifest.csv")))}
    orig_files = {f for _, f, _, _ in list_images(a.orig)}
    rec = []
    for sp, f, ip, lp in list_images(a.new):
        im = Image.open(ip)
        rec.append(dict(split=sp, file=f, img=ip, lbl=lp, src=source_name(f), sid=stock_id(source_name(f)),
                        md5=md5(ip), dh=dhash(im), ph=phash(im), th=thumb(im), group=man[f]["group_id"]))
    new_files = {r["file"] for r in rec}
    check("Same 688 images, none added/removed (no offline augmentation)", new_files == orig_files and len(rec) == len(orig_files), f"{len(rec)} images")
    check("Each image is in exactly one split", len(new_files) == len(rec), "no filename appears twice")

    # ---- 3. labels ------------------------------------------------------------------------
    stats = {s: Counter() for s in SPLITS}
    bad = []
    for r in rec:
        s = stats[r["split"]]
        s["images"] += 1
        if not os.path.exists(r["lbl"]):
            bad.append(("missing label", r["file"])); continue
        s["labels"] += 1
        L = [l.split() for l in open(r["lbl"]) if l.strip()]
        if not L: s["empty_labels"] += 1
        for t in L:
            if len(t) != 5 or t[0] != "0" or not all(0 <= float(v) <= 1 for v in t[1:]) or float(t[3]) <= 0 or float(t[4]) <= 0:
                bad.append(("bad line", r["file"]))
            s["boxes"] += 1
    check("Every image has a label file; all lines are 5-column YOLO, class 0, normalised", not bad, f"{len(bad)} problems")
    orig_boxes = sum(1 for _, _, _, lp in list_images(a.orig) for l in open(lp) if len(l.split()) == 5)
    new_boxes = sum(s["boxes"] for s in stats.values())
    check("All valid original box annotations preserved", orig_boxes == new_boxes, f"{orig_boxes} valid boxes before, {new_boxes} after")
    # per-image mapping identical
    same = True
    for r in rec:
        o = [m for m in list_images(a.orig) if m[1] == r["file"]][0] if False else None
    orig_lbl = {f: lp for _, f, _, lp in list_images(a.orig)}
    for r in rec:
        o = sorted(l.strip() for l in open(orig_lbl[r["file"]]) if len(l.split()) == 5)
        n = sorted(l.strip() for l in open(r["lbl"]) if l.strip())
        same &= len(o) == len(n)
    check("Per-image box count identical to the original (valid lines)", same)

    # ---- 4. leakage ---------------------------------------------------------------------------
    sets = {s: [r for r in rec if r["split"] == s] for s in SPLITS}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for key, label in (("group", "source-group id"), ("src", "Roboflow source name"), ("sid", "stock-photo id"), ("md5", "exact file hash")):
        shared = {p: {r[key] for r in sets[p[0]] if r[key]} & {r[key] for r in sets[p[1]] if r[key]} for p in pairs}
        n = sum(len(v) for v in shared.values())
        check(f"Zero leakage by {label}", n == 0, "; ".join(f"{p[0]}/{p[1]}: {len(v)}" for p, v in shared.items()))

    # perceptual cross-split scan (independent hash + looser thresholds than used for grouping)
    PH = np.array([r["ph"] for r in rec]); DH = np.array([r["dh"] for r in rec]); TH = np.array([r["th"] for r in rec])
    sp_arr = np.array([r["split"] for r in rec]); C = TH @ TH.T
    min_ph = min_dh = 99; max_c = -1; hits = []
    for i in range(len(rec)):
        diff = sp_arr != sp_arr[i]
        ph = (PH != PH[i]).sum(1); dh = (DH != DH[i]).sum(1)
        min_ph = min(min_ph, ph[diff].min()); min_dh = min(min_dh, dh[diff].min()); max_c = max(max_c, C[i][diff].max())
        for j in np.where(diff & ((ph <= LOOSE_PHASH) | (dh <= LOOSE_DHASH) | (C[i] >= LOOSE_CORR)))[0]:
            if j > i: hits.append((rec[i]["file"][:45], rec[i]["split"], rec[j]["file"][:45], rec[j]["split"], int(ph[j]), int(dh[j]), round(float(C[i][j]), 3)))
    check(f"Zero cross-split near-duplicates (pHash<={LOOSE_PHASH}, dHash<={LOOSE_DHASH} or corr>={LOOSE_CORR})", not hits,
          f"closest cross-split pair: pHash dist {min_ph}, dHash dist {min_dh}, max corr {max_c:.3f}")
    for h in hits[:10]: print("   hit:", h)

    # SIFT cross-split scan, deliberately looser than the grouping rule (>=8 inliers instead of >=12)
    # (verification uses every image of every source)
    rep = defaultdict(list)
    for r in sorted(rec, key=lambda r: r["md5"]):
        rep[r["src"]].append(r["img"])
    spl = {r["src"]: r["split"] for r in rec}
    table = sift_pair_table(rep)
    sift_hits = []; max_in = 0
    for (ka, kb), cands in table.items():
        if spl[ka] == spl[kb]:
            continue
        if cands:
            max_in = max(max_in, max(n for n, _ in cands))
        if is_link(cands, min_in=8, min_spread=0.03, big_in=8, big_spread=0.03):
            sift_hits.append((ka[:40], spl[ka], kb[:40], spl[kb], max(n for n, _ in cands)))
    sift_hits.sort(key=lambda h: -h[4])
    reviewed = [h for h in sift_hits if any({h[0].startswith(x), h[2].startswith(y)} == {True} or {h[0].startswith(y), h[2].startswith(x)} == {True} for x, y in REVIEWED_FALSE_POSITIVES)]
    sift_hits = [h for h in sift_hits if h not in reviewed]
    check("Zero cross-split matches by SIFT+RANSAC (any >=8-inlier match, any spread)", not sift_hits, f"highest cross-split inlier count: {max_in}; {len(reviewed)} flagged pair(s) manually reviewed = different photos (shared watermark box only)")
    for h in sift_hits[:10]: print("   sift hit:", h)

    # ---- 5. report ---------------------------------------------------------------------------------
    groups = {s: {r["group"] for r in sets[s]} for s in SPLITS}
    srcs = {s: {r["src"] for r in sets[s]} for s in SPLITS}
    tot = len(rec)
    out = ["# Dataset report - source-grouped split\n",
           "## Final counts\n",
           "| Split | Images | % | Labels (files) | Boxes | Background (empty) | Source groups | Unique source names |",
           "|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        st = stats[s]
        out.append(f"| {s} | {st['images']} | {100*st['images']/tot:.1f}% | {st['labels']} | {st['boxes']} | {st['empty_labels']} | {len(groups[s])} | {len(srcs[s])} |")
    out.append(f"| **total** | **{tot}** | 100% | {sum(st['labels'] for st in stats.values())} | {new_boxes} | {sum(st['empty_labels'] for st in stats.values())} | {sum(len(g) for g in groups.values())} | {sum(len(g) for g in srcs.values())} |")
    out += ["", "## Verification checks\n", "| Check | Result | Detail |", "|---|---|---|"] + lines
    open(os.path.join(a.new, "dataset_report.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out[:12]))
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
