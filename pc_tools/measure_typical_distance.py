"""คำนวณระยะทางจริงที่ ST ใช้เก็บข้อมูลแต่ละคลาส (จากข้อมูลจริง ไม่ใช่ประมาณ)
เพื่อรู้ว่าตอนทดสอบ ควรวางมือระยะเท่าไหร่ต่อท่า

วิธีรัน:
    python measure_typical_distance.py --data-dir C:\\ds
"""
import argparse
from pathlib import Path

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]
VALID_STATUS = {5, 9}

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True, type=Path)
args = ap.parse_args()

nearest_by_class: dict[str, list[float]] = {c: [] for c in CLASSES}

npz_files = sorted(args.data_dir.rglob("*.npz"))
print(f"กำลังอ่าน {len(npz_files)} ไฟล์ ...")

for fp in npz_files:
    class_folder = fp.parents[2].name
    if class_folder not in CLASSES:
        continue
    with np.load(fp, allow_pickle=True) as z:
        zone_data = z["zone_data"]  # (4, 64, N_frames)
    status = zone_data[0]      # (64, N)
    distance = zone_data[3]    # (64, N)
    valid_mask = np.isin(status, list(VALID_STATUS))
    for f in range(zone_data.shape[-1]):
        d = distance[:, f][valid_mask[:, f]]
        if d.size:
            nearest_by_class[class_folder].append(float(d.min()))

print(f"\n{'Class':12s} {'median':>8s} {'mean':>8s} {'p10':>8s} {'p90':>8s} {'n':>8s}")
for c in CLASSES:
    vals = np.array(nearest_by_class[c])
    if vals.size == 0:
        print(f"{c:12s}  ไม่มีข้อมูล")
        continue
    print(f"{c:12s} {np.median(vals):8.1f} {vals.mean():8.1f} "
          f"{np.percentile(vals,10):8.1f} {np.percentile(vals,90):8.1f} {vals.size:8d}")
