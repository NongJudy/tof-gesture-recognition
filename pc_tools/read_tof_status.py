"""read_tof_status.py

รับข้อมูลจากบอร์ด STM32 ทั้งระยะทาง (บรรทัด F) และค่าสถานะ (บรรทัด S)
จับคู่กันด้วยเลขเฟรม แล้วบันทึกเป็น CSV ไฟล์เดียว 129 คอลัมน์

    frame, z0..z63 (ระยะ mm), s0..s63 (สถานะ)

ความหมายของสถานะ (UM3109 ตารางที่ 4)
    5        วัดได้ถูกต้อง เชื่อได้ 100%
    6 หรือ 9 เชื่อได้ 50%
    1        แสงสะท้อนอ่อนเกินไป
    3        ค่าแกว่งเกินเกณฑ์
    255      ไม่เจอเป้าหมาย

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python read_tof_status.py
กด Ctrl+C เพื่อหยุด จะสรุปสถิติสถานะให้อัตโนมัติ
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from typing import Dict, List

import serial

PORT = "COM3"
BAUD = 460800
ZONES = 64

# สถานะที่ถือว่าใช้ได้ ตาม UM3109 (5 = 100%, 6 และ 9 = 50%)
STATUS_MEANING: Dict[int, str] = {
    5: "วัดได้ถูกต้อง 100%",
    6: "เชื่อได้ 50%",
    9: "เชื่อได้ 50%",
    1: "แสงสะท้อนอ่อนเกินไป",
    2: "สัญญาณรบกวนสูง",
    3: "ค่าแกว่งเกินเกณฑ์",
    4: "เป้านอกระยะ",
    255: "ไม่เจอเป้าหมาย",
}


def parse(text: str, tag: str) -> tuple[int, List[int]] | None:
    """แยกบรรทัด F หรือ S คืน (เลขเฟรม, ค่า 64 ตัว) ถ้าไม่ครบคืน None"""
    if not text.startswith(tag + ","):
        return None
    parts = text.split(",")
    if len(parts) != ZONES + 2:
        return None
    try:
        return int(parts[1]), [int(v) for v in parts[2:]]
    except ValueError:
        return None


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tof_status_{stamp}.csv"

    print(f"กำลังเชื่อมต่อ {PORT} ที่ {BAUD} baud ...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("เชื่อมต่อสำเร็จ")
    print(f"บันทึกลงไฟล์: {filename}")
    print("กด Ctrl+C เพื่อหยุด\n")

    header = (["frame"]
              + [f"z{i}" for i in range(ZONES)]
              + [f"s{i}" for i in range(ZONES)])

    pending: Dict[int, List[int]] = {}      # เก็บบรรทัด F ที่ยังรอบรรทัด S
    status_count: Counter = Counter()
    saved = 0

    fh = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(header)

    try:
        while True:
            text = ser.readline().decode(errors="ignore").strip()

            got = parse(text, "F")
            if got:
                pending[got[0]] = got[1]
                # กันหน่วยความจำบวม ถ้าบรรทัด S หายไปนาน
                if len(pending) > 50:
                    pending.pop(min(pending))
                continue

            got = parse(text, "S")
            if not got:
                continue

            frame_no, status = got
            dist = pending.pop(frame_no, None)
            if dist is None:            # ไม่มีคู่ ข้ามไป
                continue

            writer.writerow([frame_no] + dist + status)
            status_count.update(status)
            saved += 1

            if saved % 15 == 0:
                bad = sum(v for k, v in status_count.items() if k != 5)
                total = sum(status_count.values())
                print(f"บันทึกแล้ว {saved:4d} เฟรม | "
                      f"สถานะไม่ใช่ 5 : {bad}/{total} = {bad/total*100:.2f}%")

    except KeyboardInterrupt:
        print("\n\nหยุดการทำงาน")

    finally:
        fh.close()
        ser.close()
        print(f"บันทึกทั้งหมด {saved} เฟรม")
        print(f"ไฟล์: {filename}")

        if status_count:
            total = sum(status_count.values())
            print(f"\n===== สรุปค่าสถานะ (รวม {total} ค่า) =====")
            for code, n in status_count.most_common():
                name = STATUS_MEANING.get(code, "ไม่ทราบความหมาย")
                print(f"  สถานะ {code:3d} : {n:7d} ครั้ง "
                      f"({n/total*100:6.3f}%)  {name}")


if __name__ == "__main__":
    main()
