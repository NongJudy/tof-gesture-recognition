# compare_gestures.py
# แยกเฟรมที่มีมือออกเป็นกลุ่ม แล้วเทียบว่ากำ/แบ ต่างกันไหม

import csv, glob, os

files = glob.glob("tof_test_*.csv")
FILENAME = max(files, key=os.path.getmtime)

rows = []
with open(FILENAME, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        if r:
            rows.append([int(x) for x in r])

print("=" * 60)
print(f"ไฟล์: {FILENAME}  ({len(rows)} เฟรม)")
print("=" * 60)

# ----- คำนวณ 2 ค่าสำคัญของแต่ละเฟรม -----
stats = []
for r in rows:
    v = r[1:]
    near = sum(1 for x in v if x < 800)      # จำนวน zone ที่มีมือบัง (ใกล้กว่า 80 ซม.)
    avg_near = sum(x for x in v if x < 800) / near if near else 0
    stats.append((r[0], near, avg_near, v))

# ----- แสดงว่าแต่ละเฟรมมีมือบังกี่ zone (ดูรูปแบบการสลับท่า) -----
print("\n[1] จำนวน zone ที่มีมือบัง (ทุกๆ 10 เฟรม)")
print("    เฟรม  : zone ที่ถูกบัง")
for i in range(0, len(stats), 10):
    fno, near, _, _ = stats[i]
    bar = "#" * (near // 2)                   # กราฟแท่งง่ายๆ
    print(f"    {fno} : {near:3d}  {bar}")

# ----- จัดกลุ่ม: บังเยอะ = แบมือ / บังน้อย = กำมือ -----
hand = [s for s in stats if s[1] >= 10]       # เฉพาะเฟรมที่มีมือจริง
if hand:
    max_zone = max(s[1] for s in hand)
    min_zone = min(s[1] for s in hand)
    print(f"\n[2] เฟรมที่มีมือ : {len(hand)} เฟรม")
    print(f"    บังมากสุด : {max_zone} zones  (น่าจะแบมือ)")
    print(f"    บังน้อยสุด: {min_zone} zones  (น่าจะกำมือ)")

    def show(s, title):
        fno, near, avgn, v = s
        print(f"\n--- {title} ---")
        print(f"    frame {fno} | บัง {near} zones | ระยะเฉลี่ยของมือ {avgn:.0f} mm")
        for i in range(8):
            print("    " + "".join(f"{v[i*8+j]:6d}" for j in range(8)))

    show(max(hand, key=lambda s: s[1]), "บังพื้นที่มากสุด (แบมือ?)")
    show(min(hand, key=lambda s: s[1]), "บังพื้นที่น้อยสุด (กำมือ?)")

print("\n" + "=" * 60)