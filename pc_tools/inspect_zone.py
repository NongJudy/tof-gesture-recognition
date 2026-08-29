"""inspect_zone.py

ตรวจช่องใดช่องหนึ่งอย่างละเอียด ว่าทำไมค่าเฉลี่ยของสองเงื่อนไขต่างกัน
ใช้ต่อจาก compare_modes.py เมื่อพบช่องที่ต่างผิดปกติ

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python inspect_zone.py 60            # ตรวจช่องที่ 60 จาก 2 ไฟล์ล่าสุด
    python inspect_zone.py 60 a.csv b.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from typing import List

ZONES = 64


def load_zone(path: str, z: int) -> List[int]:
    """ดึงค่าของช่องที่ z จากทุกเฟรมในไฟล์"""
    out: List[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if row and len(row) >= ZONES + 1:
                try:
                    out.append(int(row[1 + z]))
                except ValueError:
                    pass
    return out


def describe(name: str, v: List[int]) -> None:
    med = st.median(v)
    # นับค่าที่ห่างจากค่ากลางเกิน 10% ถือว่าโดดผิดปกติ
    far = [x for x in v if abs(x - med) > med * 0.10]
    print(f"  {name}: n={len(v):4d}  เฉลี่ย {st.mean(v):8.2f}  ค่ากลาง {med:8.1f}  "
          f"sd {st.stdev(v):6.2f}  ต่ำสุด {min(v):5d}  สูงสุด {max(v):5d}")
    print(f"      ค่าที่โดดออกจากค่ากลางเกิน 10% : {len(far)} ค่า", end="")
    print(f"  ตัวอย่าง {sorted(far)[:6]}" if far else "")


def histogram(v: List[int], bins: int = 10) -> None:
    """แสดงการกระจายแบบหยาบ ๆ ดูว่าค่ากระจุกที่เดียวหรือแยกเป็นสองกลุ่ม"""
    lo, hi = min(v), max(v)
    if hi == lo:
        print("      ค่าทั้งหมดเท่ากันหมด")
        return
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in v:
        counts[min(bins - 1, int((x - lo) / w))] += 1
    top = max(counts)
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bar = "#" * max(1, int(c / top * 34))
        print(f"      {lo + i*w:7.0f}-{lo + (i+1)*w:7.0f} mm | {c:4d} {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zone", type=int, help="หมายเลขช่อง 0-63")
    ap.add_argument("before", nargs="?")
    ap.add_argument("after", nargs="?")
    a = ap.parse_args()

    if a.before and a.after:
        pa, pb = a.before, a.after
    else:
        files = sorted(glob.glob("tof_test_*.csv"), key=os.path.getmtime)
        if len(files) < 2:
            raise SystemExit("ต้องมีไฟล์ tof_test_*.csv อย่างน้อย 2 ไฟล์")
        pa, pb = files[-2], files[-1]

    z = a.zone
    print("=" * 74)
    print(f"ตรวจช่องที่ {z}  (แถว {z // 8} คอลัมน์ {z % 8})")
    print("=" * 74)

    va, vb = load_zone(pa, z), load_zone(pb, z)

    print(f"\n[1] สถิติของช่องที่ {z}")
    describe("A", va)
    describe("B", vb)

    print(f"\n[2] การกระจายของค่า — เงื่อนไข A")
    histogram(va)
    print(f"\n[3] การกระจายของค่า — เงื่อนไข B")
    histogram(vb)

    # เทียบกับช่องข้างเคียง เพื่อดูว่าเป็นเฉพาะช่องนี้จริงไหม
    nb = [i for i in (z - 8, z - 1, z + 1, z + 8) if 0 <= i < ZONES]
    print(f"\n[4] ช่องข้างเคียง (บน ซ้าย ขวา ล่าง)")
    for i in nb:
        ma = st.mean(load_zone(pa, i))
        mb = st.mean(load_zone(pb, i))
        print(f"  ช่อง {i:2d}: A {ma:8.2f}  B {mb:8.2f}  ต่าง {mb-ma:+7.2f} mm")

    print("\n" + "=" * 74)
    sda, sdb = st.stdev(va), st.stdev(vb)
    print("อ่านผลยังไง")
    print(f"  sd ของช่องนี้ = {sda:.1f} / {sdb:.1f} mm")
    if max(sda, sdb) > 30:
        print("  -> sd สูงมาก แปลว่าค่าแกว่งเองอยู่แล้ว ไม่ใช่ผลจากการเปลี่ยนโหมด")
        print("     มักเกิดกับช่องที่ส่องไปโดนขอบวัตถุ หรือพื้นผิวเอียง")
    else:
        print("  -> sd ต่ำ แปลว่าค่านิ่ง ความต่างที่เจอน่าจะมีสาเหตุจริง ต้องดูต่อ")
    print("=" * 74)


if __name__ == "__main__":
    main()
