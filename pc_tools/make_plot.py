# make_plot.py
# สร้างกราฟผลการทดลอง สำหรับใส่รายงาน

import csv, glob, os
import matplotlib.pyplot as plt

files = glob.glob("tof_test_*.csv")
FILENAME = max(files, key=os.path.getmtime)

rows = []
with open(FILENAME, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        if r:
            rows.append([int(x) for x in r])

# นับจำนวน zone ที่มีมือบัง (ระยะน้อยกว่า 80 ซม.)
counts = [sum(1 for x in r[1:] if x < 800) for r in rows]
frames = list(range(len(counts)))

# ----- วาดกราฟ -----
plt.figure(figsize=(11, 4.5))
plt.plot(frames, counts, linewidth=1.2, color="#1f77b4")

plt.xlabel("Frame number")
plt.ylabel("Number of occupied zones")
plt.title("Occupied zones over time: alternating open hand and fist at 30 cm")
plt.grid(alpha=0.3)

# เส้นแบ่งกลุ่ม 2 ท่า
plt.axhline(22, color="red", linestyle="--", linewidth=1,
            label="Separation threshold")
plt.legend()

plt.tight_layout()
plt.savefig("result_gesture_separation.png", dpi=200)
print("บันทึกกราฟแล้ว: result_gesture_separation.png")
print(f"ใช้ข้อมูลจากไฟล์: {FILENAME}  ({len(rows)} เฟรม)")