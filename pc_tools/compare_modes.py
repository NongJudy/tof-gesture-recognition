"""compare_modes.py

เทียบข้อมูลระยะทางจาก 2 เงื่อนไข (เช่น polling กับ INT) ว่าเหมือนกันหรือไม่
ใช้ยืนยันว่าการเปลี่ยนวิธีอ่านข้อมูลไม่ได้ทำให้ข้อมูลเปลี่ยน

เงื่อนไขการใช้งาน: ทั้งสองไฟล์ต้องเก็บจาก "ฉากเดียวกัน" ไม่ขยับเซ็นเซอร์ระหว่างรอบ

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python compare_modes.py ไฟล์ก่อน.csv ไฟล์หลัง.csv
    python compare_modes.py                # ใช้ 2 ไฟล์ล่าสุดอัตโนมัติ
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from typing import List, Tuple

ZONES = 64

# เกณฑ์ตัดสิน: ค่าเฉลี่ยรายช่องต่างกันไม่เกินกี่ mm ถือว่าเหมือนกัน
# ตั้งจากความละเอียดของเซ็นเซอร์เอง (VL53L8CX ระบุ +/- 1 cm ที่ระยะใกล้)
THRESHOLD_MM = 10.0


def load(path: str) -> Tuple[List[int], List[List[int]]]:
    """อ่าน CSV คืน (เลขเฟรม, ข้อมูล 64 ช่องต่อเฟรม)"""
    frames: List[int] = []
    zones: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)                      # ข้ามหัวตาราง
        for row in reader:
            if not row or len(row) < ZONES + 1:
                continue
            try:
                frames.append(int(row[0]))
                zones.append([int(v) for v in row[1:ZONES + 1]])
            except ValueError:
                continue
    if not zones:
        raise ValueError(f"อ่านข้อมูลจาก {path} ไม่ได้")
    return frames, zones


def gaps(frames: List[int]) -> int:
    """นับจุดที่เลขเฟรมขาดหาย"""
    return sum(1 for a, b in zip(frames, frames[1:]) if b != a + 1)


def per_zone_mean(zones: List[List[int]]) -> List[float]:
    """ค่าเฉลี่ยของแต่ละช่อง ตลอดทุกเฟรม"""
    return [st.mean(f[i] for f in zones) for i in range(ZONES)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", nargs="?", help="ไฟล์เงื่อนไขที่ 1")
    ap.add_argument("after", nargs="?", help="ไฟล์เงื่อนไขที่ 2")
    args = ap.parse_args()

    if args.before and args.after:
        pa, pb = args.before, args.after
    else:
        files = sorted(glob.glob("tof_test_*.csv"), key=os.path.getmtime)
        if len(files) < 2:
            raise SystemExit("ต้องมีไฟล์ tof_test_*.csv อย่างน้อย 2 ไฟล์")
        pa, pb = files[-2], files[-1]

    fa, za = load(pa)
    fb, zb = load(pb)

    print("=" * 74)
    print(f"เงื่อนไข A : {pa}")
    print(f"เงื่อนไข B : {pb}")
    print("=" * 74)

    print("\n[1] ความสมบูรณ์ของข้อมูล")
    for name, fr, z in (("A", fa, za), ("B", fb, zb)):
        print(f"  {name}: {len(z):4d} เฟรม | เลขเฟรม {fr[0]}-{fr[-1]} | "
              f"เฟรมขาดหาย {gaps(fr)}")

    ma, mb = per_zone_mean(za), per_zone_mean(zb)
    diffs = [b - a for a, b in zip(ma, mb)]
    absd = [abs(d) for d in diffs]

    print("\n[2] ค่าเฉลี่ยรวมทุกช่อง")
    print(f"  A = {st.mean(ma):8.2f} mm")
    print(f"  B = {st.mean(mb):8.2f} mm")
    print(f"  ต่างกัน {st.mean(mb) - st.mean(ma):+8.2f} mm")

    print("\n[3] เทียบรายช่อง (64 ช่อง)")
    print(f"  ต่างเฉลี่ย  {st.mean(absd):6.2f} mm")
    print(f"  ต่างมากสุด {max(absd):6.2f} mm  (ช่องที่ {absd.index(max(absd))})")
    over = [i for i, d in enumerate(absd) if d > THRESHOLD_MM]
    print(f"  ช่องที่ต่างเกิน {THRESHOLD_MM:.0f} mm : {len(over)} / {ZONES}")
    if over:
        print(f"    ช่อง: {over[:12]}")

    print("\n[4] แผนที่ความต่างรายช่อง (mm, B ลบ A)")
    for r in range(8):
        print("   " + " ".join(f"{diffs[r * 8 + c]:+7.1f}" for c in range(8)))

    print("\n" + "=" * 74)
    ok_gap = gaps(fa) == 0 and gaps(fb) == 0
    ok_diff = max(absd) <= THRESHOLD_MM
    if ok_gap and ok_diff:
        print("ผลสรุป: ผ่าน  ข้อมูลทั้งสองเงื่อนไขเหมือนกันในทางสถิติ")
        print("        การเปลี่ยนวิธีอ่านข้อมูลไม่ได้ทำให้ข้อมูลเปลี่ยน")
    else:
        print("ผลสรุป: ต้องตรวจเพิ่ม")
        if not ok_gap:
            print("        - มีเฟรมขาดหาย")
        if not ok_diff:
            print(f"        - มีช่องที่ต่างเกิน {THRESHOLD_MM:.0f} mm")
            print("        (เช็คก่อนว่าไม่ได้ขยับเซ็นเซอร์ระหว่างสองรอบ)")
    print("=" * 74)


if __name__ == "__main__":
    main()
