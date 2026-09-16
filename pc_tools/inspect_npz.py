"""ตรวจโครงสร้างไฟล์ .npz จริง 1 ไฟล์ + ชื่อโฟลเดอร์ทั้งหมด เพื่อหาว่าโค้ดโหลดข้อมูล
เข้าใจผิดตรงไหน ก่อนจะแก้ train_st_cnn2d.py ให้ถูกต้อง

วิธีรัน:
    python inspect_npz.py --data-dir C:\\ds
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True, type=Path)
args = ap.parse_args()

npz_files = sorted(args.data_dir.rglob("*.npz"))
print(f"เจอไฟล์ .npz ทั้งหมด {len(npz_files)} ไฟล์\n")

print("=== ชื่อโฟลเดอร์ 10 อันแรก (เพื่อดูรูปแบบชื่อ user) ===")
seen_dirs = []
for fp in npz_files:
    d = str(fp.parent)
    if d not in seen_dirs:
        seen_dirs.append(d)
    if len(seen_dirs) >= 10:
        break
for d in seen_dirs:
    print(" ", d)

print("\n=== เปิดไฟล์แรกดูโครงสร้างเต็ม ===")
fp = npz_files[0]
print("ไฟล์:", fp)
with np.load(fp, allow_pickle=True) as z:
    print("keys ในไฟล์:", list(z.keys()))
    for k in z.keys():
        arr = z[k]
        print(f"\n  key = '{k}'")
        print(f"    dtype = {arr.dtype}, shape = {arr.shape}")
        if arr.size <= 20:
            print(f"    ค่า = {arr}")
        else:
            print(f"    ตัวอย่างค่า (5 แรก แบน) = {arr.reshape(-1)[:5]}")

print("\n=== ลองอ่านค่า label ตามที่โค้ดเดิมสมมติไว้ ===")
with np.load(fp, allow_pickle=True) as z:
    glob_head = list(z["glob_head"])
    glob_data = z["glob_data"]
    print("glob_head:", glob_head)
    print("glob_data:", glob_data, " shape:", glob_data.shape)
