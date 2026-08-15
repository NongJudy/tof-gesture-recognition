# check_csv.py
# ตรวจสอบไฟล์ CSV ล่าสุดในโฟลเดอร์นี้ ว่าข้อมูลถูกต้องครบถ้วนไหม

import csv
import glob
import os

# ----- หาไฟล์ CSV ล่าสุดอัตโนมัติ (ไม่ต้องพิมพ์ชื่อเอง) -----
files = glob.glob("tof_test_*.csv")
if not files:
    print("ไม่พบไฟล์ tof_test_*.csv ในโฟลเดอร์นี้")
    raise SystemExit

FILENAME = max(files, key=os.path.getmtime)   # เอาไฟล์ที่แก้ไขล่าสุด

# ----- อ่านไฟล์ -----
rows = []
with open(FILENAME, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        if r:
            rows.append(r)

print("=" * 55)
print(f"ไฟล์: {FILENAME}")
print("=" * 55)

# ----- 1. โครงสร้างไฟล์ -----
print(f"\n[1] จำนวนคอลัมน์ : {len(header)}   (ควรเป็น 65)")
print(f"    หัวตาราง 5 ตัวแรก  : {header[:5]}")
print(f"    หัวตาราง 2 ตัวท้าย : {header[-2:]}")
print(f"    จำนวนแถวข้อมูล    : {len(rows)}")

# ----- 2. เลขเฟรมต่อเนื่องไหม -----
frames = [int(r[0]) for r in rows]
gaps = sum(1 for i in range(1, len(frames)) if frames[i] != frames[i-1] + 1)

print(f"\n[2] เลขเฟรม : {frames[0]} ถึง {frames[-1]}")
print(f"    จุดที่เฟรมขาดหาย : {gaps}   (ควรเป็น 0)")

# ----- 3. ค่าเฉลี่ยแต่ละเฟรม -----
averages = []
for r in rows:
    values = [int(v) for v in r[1:]]
    averages.append(sum(values) / len(values))

hi = max(averages)
lo = min(averages)
print(f"\n[3] ค่าเฉลี่ยสูงสุด : {hi:7.0f} mm   (น่าจะตอนไม่มีมือ)")
print(f"    ค่าเฉลี่ยต่ำสุด : {lo:7.0f} mm   (น่าจะตอนมีมือบัง)")
print(f"    ความต่าง        : {hi - lo:7.0f} mm")

# ----- 4. นับเฟรมที่มีมือ -----
threshold = (hi + lo) / 2
hand = sum(1 for a in averages if a < threshold)
print(f"\n[4] เส้นแบ่ง : {threshold:.0f} mm")
print(f"    เฟรมที่น่าจะ 'มีมือ'   : {hand}")
print(f"    เฟรมที่น่าจะ 'ไม่มีมือ' : {len(rows) - hand}")

# ----- 5. แสดงตาราง 8x8 ตัวอย่าง -----
def show(row, title):
    print(f"\n--- {title} (frame {row[0]}) ---")
    v = [int(x) for x in row[1:]]
    for i in range(8):
        print("".join(f"{v[i*8+j]:6d}" for j in range(8)))

show(rows[averages.index(hi)], "ค่าเฉลี่ยสูงสุด = ไม่มีมือ")
show(rows[averages.index(lo)], "ค่าเฉลี่ยต่ำสุด = มีมือบังมากสุด")

print("\n" + "=" * 55)