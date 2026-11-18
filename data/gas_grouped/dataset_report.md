# Dataset report - source-grouped split

## Final counts

| Split | Images | % | Labels (files) | Boxes | Background (empty) | Source groups | Unique source names |
|---|---|---|---|---|---|---|---|
| train | 482 | 70.1% | 482 | 489 | 3 | 46 | 124 |
| valid | 103 | 15.0% | 103 | 103 | 0 | 30 | 35 |
| test | 103 | 15.0% | 103 | 103 | 0 | 31 | 40 |
| **total** | **688** | 100% | 688 | 695 | 3 | 107 | 199 |

## Verification checks

| Check | Result | Detail |
|---|---|---|
| Original backup unchanged (sha256 of all files) | PASS | all checksums match |
| Same 688 images, none added/removed (no offline augmentation) | PASS | 688 images |
| Each image is in exactly one split | PASS | no filename appears twice |
| Every image has a label file; all lines are 5-column YOLO, class 0, normalised | PASS | 0 problems |
| All valid original box annotations preserved | PASS | 695 valid boxes before, 695 after |
| Per-image box count identical to the original (valid lines) | PASS |  |
| Zero leakage by source-group id | PASS | train/valid: 0; train/test: 0; valid/test: 0 |
| Zero leakage by Roboflow source name | PASS | train/valid: 0; train/test: 0; valid/test: 0 |
| Zero leakage by stock-photo id | PASS | train/valid: 0; train/test: 0; valid/test: 0 |
| Zero leakage by exact file hash | PASS | train/valid: 0; train/test: 0; valid/test: 0 |
| Zero cross-split near-duplicates (pHash<=8, dHash<=8 or corr>=0.9) | PASS | closest cross-split pair: pHash dist 14, dHash dist 12, max corr 0.883 |
| Zero cross-split matches by SIFT+RANSAC (any >=8-inlier match, any spread) | PASS | highest cross-split inlier count: 12; 1 flagged pair(s) manually reviewed = different photos (shared watermark box only) |
