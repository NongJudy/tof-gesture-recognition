# read_tof.py
# รับข้อมูล ToF จากบอร์ด STM32 -> แสดงตาราง 8x8 -> เซฟลงไฟล์ CSV

import serial
import csv
from datetime import datetime

# ===== ค่าตั้งต้น =====
PORT  = "COM3"
BAUD  = 460800
ZONES = 64          # จำนวน zone (8x8)

# ===== ตั้งชื่อไฟล์ตามวันเวลา (กันเขียนทับไฟล์เก่า) =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = f"tof_test_{timestamp}.csv"

print(f"กำลังเชื่อมต่อ {PORT} ที่ {BAUD} baud ...")
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"เชื่อมต่อสำเร็จ!")
print(f"บันทึกลงไฟล์: {filename}")
print("กด Ctrl+C เพื่อหยุด\n")

saved_count = 0     # นับจำนวนเฟรมที่บันทึกแล้ว

# เปิดไฟล์เตรียมเขียน (newline="" กันบรรทัดว่างแทรกบน Windows)
csv_file = open(filename, "w", newline="", encoding="utf-8")
writer   = csv.writer(csv_file)

# ----- เขียนหัวตาราง: frame, z0, z1, ..., z63 -----
header = ["frame"] + [f"z{i}" for i in range(ZONES)]
writer.writerow(header)

try:
    while True:
        line = ser.readline()
        text = line.decode(errors="ignore").strip()

        if not text.startswith("F,"):      # ข้ามบรรทัดที่ไม่ใช่ข้อมูล
            continue

        parts = text.split(",")
        if len(parts) != ZONES + 2:        # ต้องมี F + เลขเฟรม + 64 ค่า
            continue                       # ข้อมูลไม่ครบ -> ข้ามไป

        frame_no = int(parts[1])
        values   = [int(v) for v in parts[2:]]

        # ----- บันทึกลงไฟล์ -----
        writer.writerow([frame_no] + values)
        saved_count += 1

        # ----- แสดงบนจอทุก 15 เฟรม (ไม่ให้จอรกเกินไป) -----
        if saved_count % 15 == 0:
            print(f"\n===== Frame {frame_no}  (บันทึกแล้ว {saved_count} เฟรม) =====")
            for row in range(8):
                line_out = ""
                for col in range(8):
                    line_out += f"{values[row * 8 + col]:6d}"
                print(line_out)

except KeyboardInterrupt:
    print("\nหยุดการทำงาน")

finally:
    csv_file.close()                       # ปิดไฟล์ให้เรียบร้อย (สำคัญมาก!)
    ser.close()
    print(f"บันทึกทั้งหมด {saved_count} เฟรม")
    print(f"ไฟล์: {filename}")