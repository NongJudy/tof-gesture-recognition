"""อ่านข้อมูล ToF สดจากบอร์ดผ่าน serial (F,/S,/G, lines) ส่งเข้าโมเดล production
ที่เทรนไว้ ทำนายท่ามือแบบเรียลไทม์ พร้อม majority vote กรอง error สุ่ม

ก่อนรันสคริปต์นี้ ต้องทำให้ครบก่อน:
  1. บอร์ด build+flash ด้วย MY_TOF_USE_4X4=1, MY_TOF_TIMING_MODE=0
  2. ตรวจสอบแล้วว่า G, line (signal) มีค่าสมเหตุสมผล (ไม่ใช่ 0 หรือค่าคงที่ตลอด)
  3. เทรน production model ไว้แล้ว (train_st_cnn2d.py --split production)

วิธีรัน:
    pip install pyserial tensorflow --break-system-packages
    python live_predict.py --port COM3 --model results_phase2\\production_model.keras

จุดที่อาจพลาด (ยังไม่เคยทดสอบกับบอร์ดจริง — เทสต์แค่ด้วยข้อมูลจำลอง):
  - รูปแบบบรรทัด F,/S,/G, สมมติตามที่บันทึกไว้ใน PROJECT_FACTS §4 ยังไม่เคยเห็น
    ค่าจริงจาก G, line เลย ถ้ารูปแบบจริงต่างจากนี้ ต้องแก้ parse_line()
  - baud rate ตั้งตาม PROJECT_FACTS §1.3 (460800) ถ้าบอร์ดตั้งไว้คนละค่าต้องแก้ --baud
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    raise SystemExit("ต้องติดตั้งก่อน: pip install pyserial --break-system-packages")

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit("ต้องติดตั้งก่อน: pip install tensorflow --break-system-packages")

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]
MAX_DISTANCE = 400.0
BACKGROUND_DISTANCE = 120.0
VALID_STATUS = {5, 9}
DIST_MEAN, DIST_STD = 295.0, 196.0
SIG_MEAN, SIG_STD = 281.0, 452.0


def parse_line(line: str) -> tuple[str, int, list[float]] | None:
    """แยกบรรทัด 'F,12,105,110,...' หรือ 'S,12,5,5,...' หรือ 'G,12,320,318,...'
    คืนค่า (ชนิดบรรทัด, เลขเฟรม, ค่าตามจำนวนโซนจริง) หรือ None ถ้า parse ไม่ได้
    ไม่ hardcode จำนวนโซน — เช็คว่าตรง 64 (โหมด 8x8) หรือไม่ ทำที่ main() แทน
    เพื่อให้ error message ชัดเจนว่า 'โหมดผิด' ไม่ใช่ 'parse ไม่ได้'
    """
    parts = line.strip().split(",")
    if len(parts) < 2 or parts[0] not in ("F", "S", "G"):
        return None
    try:
        frame_no = int(parts[1])
        values = [float(v) for v in parts[2:]]
    except ValueError:
        return None
    if len(values) not in (16, 64):  # 4x4 หรือ 8x8 เท่านั้นที่เป็นไปได้จริง
        return None
    return parts[0], frame_no, values


def preprocess(distance: np.ndarray, status: np.ndarray, signal: np.ndarray) -> np.ndarray | None:
    """เตรียมข้อมูล 1 เฟรมให้ตรงกับ preprocessing ตอนเทรนเป๊ะ (ใช้ค่าคงที่ชุดเดียวกัน)"""
    valid_mask = np.isin(status, list(VALID_STATUS))
    nearest = distance[valid_mask]
    if nearest.size == 0 or nearest.min() > MAX_DISTANCE:
        return None
    d = distance.copy()
    s = signal.copy()
    bad = ~valid_mask | (d > nearest.min() + BACKGROUND_DISTANCE)
    d[bad] = 4000.0
    s[bad] = 0.0
    d_norm = (d - DIST_MEAN) / DIST_STD
    s_norm = (s - SIG_MEAN) / SIG_STD
    return np.stack([d_norm.reshape(8, 8), s_norm.reshape(8, 8)], axis=-1)


def read_frames(ser: serial.Serial):
    """generator: อ่านทีละบรรทัดจาก serial ประกอบ F,/S,/G, ของเฟรมเดียวกันให้ครบ
    แล้ว yield เป็น (frame_no, distance[64], status[64], signal[64])
    """
    pending: dict[int, dict[str, list[float]]] = {}
    warned_wrong_mode = False
    while True:
        raw = ser.readline().decode("utf-8", errors="ignore")
        parsed = parse_line(raw)
        if parsed is None:
            continue
        kind, frame_no, values = parsed
        if len(values) == 16 and not warned_wrong_mode:
            print("\n[คำเตือน] บอร์ดกำลังส่งข้อมูลโหมด 4x4 (16 โซน) แต่โมเดลต้องการ "
                  "8x8 (64 โซน) — ไปแก้ MY_TOF_USE_4X4 เป็น 0 ใน my_tof.h แล้ว "
                  "build+flash ใหม่ก่อนครับ จะข้ามข้อมูลชุดนี้ไปเรื่อยๆ จนกว่าจะแก้\n")
            warned_wrong_mode = True
        if len(values) != 64:
            continue  # ข้ามเฟรมโหมด 4x4 ไป รอจนกว่าจะเป็น 8x8 จริง
        slot = pending.setdefault(frame_no, {})
        slot[kind] = values
        if all(k in slot for k in ("F", "S", "G")):
            yield frame_no, np.array(slot["F"]), np.array(slot["S"]), np.array(slot["G"])
            del pending[frame_no]
        # กันหน่วยความจำบวมถ้าเฟรมไหนมาไม่ครบ (สาย/ขาดหาย)
        stale = [k for k in pending if frame_no - k > 30]
        for k in stale:
            del pending[k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="เช่น COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--vote-window", type=int, default=15,
                     help="จำนวนเฟรมที่ใช้ majority vote ก่อนฟันธงคำตอบ "
                          "(15 เฟรม ที่ 60 Hz = 0.25 วินาที)")
    args = ap.parse_args()

    print(f"กำลังโหลดโมเดลจาก {args.model} ...")
    model = tf.keras.models.load_model(args.model)

    print(f"เปิด serial port {args.port} @ {args.baud} baud ...")
    ser = serial.Serial(args.port, args.baud, timeout=1)

    vote_buffer: deque[int] = deque(maxlen=args.vote_window)
    print(f"พร้อมแล้ว — ทำท่ามือได้เลย (majority vote ทุก {args.vote_window} เฟรม, Ctrl+C เพื่อหยุด)")

    try:
        for frame_no, distance, status, signal in read_frames(ser):
            x = preprocess(distance, status, signal)
            if x is None:
                continue  # ไม่มีมือในเฟรมนี้ (ไกลเกิน MAX_DISTANCE ทั้งเฟรม)
            pred = model.predict(x[np.newaxis, ...], verbose=0)[0]
            class_idx = int(np.argmax(pred))
            vote_buffer.append(class_idx)

            if len(vote_buffer) == args.vote_window:
                winner, count = Counter(vote_buffer).most_common(1)[0]
                agreement = count / args.vote_window
                print(f"[เฟรม {frame_no}] โหวต {args.vote_window} เฟรม -> "
                      f"{CLASSES[winner]}  (เห็นตรงกัน {agreement*100:.0f}%, "
                      f"เฟรมล่าสุดทายว่า {CLASSES[class_idx]} มั่นใจ {pred[class_idx]*100:.0f}%)")
    except KeyboardInterrupt:
        print("\nหยุดแล้ว")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
