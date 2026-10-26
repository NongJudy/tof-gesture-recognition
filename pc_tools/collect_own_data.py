"""เก็บข้อมูลท่ามือ 7 ท่าจากบอร์ดของเราเอง (Phase 4/5) ในรูปแบบเดียวกับ
ST_VL53L8CX_handposture_dataset เพื่อให้ train_st_cnn2d.py / st_dataset.py อ่านได้

ก่อนรัน (ตรวจให้ครบ — PROJECT_FACTS §5.1e):
  1. บอร์ด flash ด้วย MY_TOF_USE_4X4=0 (8x8), MY_TOF_TIMING_MODE=0, MY_TOF_FAST_READ=1
     (FAST_READ=1 ทำให้บรรทัด S, เป็นค่า status ดิบของ ST คือ 5 = ใช้ได้
      ถ้า FAST_READ=0 ค่าจะผ่าน BSP remap (0 = ใช้ได้) และสคริปต์นี้จะเตือนตอนตรวจก่อนเก็บ)
  2. ปิดโปรแกรมอื่นที่เปิด COM port ค้างไว้ (PuTTY, live_predict.py ฯลฯ)
  3. ที่ว่างโล่ง ไม่มีผนังหรือสิ่งของในมุมมองเซ็นเซอร์ภายใน ~40 ซม.
  4. ตอนโปรแกรมขึ้น "ตรวจบอร์ด" ให้ยื่นมือค้างไว้หน้าเซ็นเซอร์ก่อน แล้วค่อยกด Enter
     (ถ้าไม่มีอะไรในระยะ การตรวจ status อาจหยุดสคริปต์เพราะเข้าใจผิดว่า status เพี้ยน)
  5. เปลี่ยนแป้นพิมพ์เป็นภาษาอังกฤษก่อนเริ่ม (ตอบ y/r/s) — แป้นไทยก็รับ แต่อังกฤษชัวร์กว่า
  6. แนะนำให้รันใน Windows Terminal (แสดงภาษาไทยบรรทัดเดียวได้เรียบร้อยกว่า cmd แบบเก่า)

ขั้นตอนต่อรอบ (เพิ่ม 9 ต.ค. 2026):
  กด Enter → จอแสดง "ระยะมือ" บรรทัดเดียว ตัวเลขเปลี่ยนอยู่กับที่ + ช่วงที่ต้องการ (เช่น 220-320 มม.)
  → ทำท่าให้เสร็จก่อน แล้วค่อยยื่นมือเข้าช่วง → ค้างในช่วงต่อเนื่อง --hold-s วินาที (ค่าเริ่มต้น 3)
    ถ้าหลุดช่วงระหว่างนับ นับใหม่จาก 0
  → ครบแล้วเริ่มเก็บเอง (มีเสียงบี๊ป) --seconds วินาที (ค่าเริ่มต้น 20)
    ถ้ามือหลุดช่วงติดกัน >= --abort-frames เฟรม (ค่าเริ่มต้น 8 ≈ 0.5 วิ) ทิ้งทั้งรอบแล้วทำใหม่
    หลุดสั้นกว่านั้น (ค่าแกว่ง/มือสั่น) ไม่ทิ้งเฟรม แต่นับไว้ใน out_of_window_frames

วิธีรัน (ที่ root ของ repo, ตัวอย่าง ผู้ทดลองคนที่ 5, เปิดเครื่องมาแล้ว 12 นาที):
    python pc_tools\\collect_own_data.py --port COM3 --user 5 --minutes-since-poweron 12
ตัวเลือกที่ปรับได้: --target-mm 270 --tol-mm 50 (ช่วง = 220-320) --hold-s 3 --abort-frames 8
    --arm-timeout-s 60 (รอมือเข้าช่วงนานสุด) --no-beep --no-color

ผลลัพธ์:
    <out-dir>/<ชื่อคลาส>/log__<เวลา>__User<N>__take<k>/npz/<...>.npz
    <out-dir>/<ชื่อคลาส>/log__<เวลา>__User<N>__take<k>/npz/<...>_board.csv   (v6: ผลทาย+เวลาของบอร์ดต่อเฟรม)
    <out-dir>/manifest_collect.csv   (บันทึกทุกรอบ: ผู้ทดลอง ท่า เฟรม เฟรมหาย เฟรมค้างที่ทิ้ง ระยะ ฯลฯ)

หมายเหตุรูปแบบ npz:
    zone_data (4, 64, N) ลำดับแกนแรก = [target_status, valid, signal_per_spad, distance_mm]
    glob_data (1, N)     = รหัสตัวเลขของ ST ต่อท่า (ST_LABEL) อ่านได้ทั้งด้วย st_dataset.py
    (ใช้รหัสตัวเลข) และ train_st_cnn2d.py (ใช้ชื่อโฟลเดอร์ — PROJECT_FACTS §9 #33)
    ระวัง: ลำดับดัชนีคลาสของสองสคริปต์ไม่เหมือนกัน ตอนประเมินให้แมปด้วย "ชื่อ" เสมอ (§9 #38)
    valid ตั้งเท่ากับ (status ใช้ได้) เพื่อให้ผลเหมือนใช้ status อย่างเดียว
    start_tstmp / end_tstmp = เวลาเริ่ม/จบการอ่านรอบนั้นจริง (Unix time)

ประวัติการแก้:
  - 8 ต.ค. 2026 (หลัง dry run ครั้งแรกบนบอร์ดจริง): พบว่าทุกรอบมี "เฟรมค้าง" 1-2 เฟรม
    จากช่วงก่อนกด Enter หลุดเข้ามาที่หัวไฟล์ แม้จะเรียก reset_input_buffer() แล้ว
    หลักฐาน: ได้ 76 เฟรมทุกรอบ แต่ "หาย" = 291 และ 3683 ซึ่งโตตามเวลาที่รอ
    (3683 / 15 Hz ≈ 4 นาที = เวลาตั้งแต่รอบก่อนจบ) และเฟรมแรกของรอบ FlatHand
    มีระยะ 217/215 มม. เท่ากับรอบ Fist ก่อนหน้า ส่วนเฟรมถัดไปเป็น 238-244 มม.
    แก้ 2 ชั้น: (1) อ่านทิ้ง DRAIN_S วินาทีหลังล้าง buffer (2) ถ้าใน LEAD_CHECK เฟรมแรก
    มีเลขเฟรมกระโดดเกิน STALE_GAP ให้ทิ้งทุกเฟรมก่อนจุดกระโดด และบันทึกจำนวนที่ทิ้ง
  - 8 ต.ค. 2026: คำตอบ y รับสระ "ั" ด้วย (ปุ่มเดียวกับ Y บนแป้นไทย) เพราะพิมพ์ "ัyy"
    ตอนแป้นเป็นภาษาไทยแล้วรอบนั้นถูกข้ามโดยไม่ตั้งใจ
  - 8 ต.ค. 2026: start_tstmp/end_tstmp เดิมเป็นเวลาตอนบันทึกทั้งคู่ เปลี่ยนเป็นเวลาอ่านจริง
  - 8 ต.ค. 2026: รอบที่มีคำเตือน (เช่น ระยะเกิน ±50 มม.) เลือกได้ 3 ทาง y = เก็บ,
    r = ทำรอบนี้ใหม่, s = ข้าม — เดิมตอบอย่างอื่นนอกจาก y คือข้ามไปรอบถัดไปเลย
    ทำให้ท่านั้นได้รอบไม่ครบ คำตอบที่ไม่รู้จักจะถามซ้ำ ไม่ข้ามเงียบๆ
    manifest มีคอลัมน์ attempt บอกว่าเก็บได้ในครั้งที่เท่าไหร่
  - 9 ต.ค. 2026: เปลี่ยนจาก "นับถอยหลัง 3-2-1 แล้วเก็บเลย" เป็น "รอมือเข้าช่วงระยะ ค้างครบ
    3 วิ แล้วเริ่มเก็บเอง" ตามที่ Chin กำหนด เพื่อไม่ให้มีเฟรมช่วงยื่นมือเข้า/ออกปนในข้อมูล
    และให้ระยะของทุกรอบถูกควบคุมเท่ากัน แสดงระยะสดบรรทัดเดียว (ไม่ไหลลงเป็นแถว)
    เกณฑ์ทิ้งรอบใช้ "หลุดติดกัน 8 เฟรม" ไม่ใช่ "หลุด 1 เฟรม" เพราะระยะที่วัดแกว่งเองได้
    ไม่กี่มิลลิเมตรและมือคนสั่น ถ้าใช้ 1 เฟรม รอบที่มืออยู่ขอบช่วงจะถูกทิ้งแทบทุกรอบ
    ขั้นเก็บอ่านต่อจาก stream เดียวกับขั้นรอ (ไม่ล้าง buffer ตรงกลาง) จึงไม่มีเฟรมค้าง
    และไม่ต้องเรียก drop_stale_lead() ในรอบจริง (ยังใช้ในขั้นตรวจก่อนเก็บ)
    manifest เพิ่มคอลัมน์ out_of_window_frames, window_lo_mm, window_hi_mm, hold_s, abort_frames
    → manifest ของรุ่นก่อน (เช่น data/phase4_dryrun2) ใช้ต่อไม่ได้ ต้องเป็นโฟลเดอร์ใหม่
  - 9 ต.ค. 2026 (หลัง dry run 3): จัดหน้าจอใหม่ให้อ่านง่ายและแคปใส่รายงานได้ —
    กรอบข้อมูลการเก็บตอนเริ่ม, หัวข้อแต่ละรอบ, แถบความคืบหน้า, ป้ายสถานะมีสี
    (เขียว = ในช่วง, เหลือง = ใกล้/ไกลไป, แดง = ไม่พบมือ/ทิ้งรอบ), กล่องผลต่อรอบ
    และตารางสรุปตอนจบ ; ขั้นตรวจบอร์ดเปลี่ยนเป็น "ยื่นมือก่อนแล้วกด Enter" เพราะ dry run 3
    ยื่นมือไม่ทันช่วงอ่าน 3 วิ (ขึ้นคำเตือน 0% ในช่วง 100-400 มม.) ; ตรรกะการเก็บไม่เปลี่ยน
  - 9 ต.ค. 2026: ข้อความบนหน้าจอเปลี่ยนเป็นภาษาอังกฤษทั้งหมด เพราะฟอนต์ของ Windows Terminal
    ไม่มีอักษรไทย ทำให้สระหน้า (เ แ ไ) ถูกเว้นช่องและดูไม่เรียบร้อยในรูปที่แคปใส่รายงาน
    (คำอธิบาย/คอมเมนต์ในโค้ดยังเป็นภาษาไทย ; แป้นไทย ั/พ/ห ยังใช้ตอบ y/r/s ได้เหมือนเดิม)
  - 15 ต.ค. 2026 (v5, §9 #53): หลังทุกรอบที่เก็บได้ จะถามคนคุมเสมอ
    "Posture correct the whole time? y/r/s" — เดิมรอบที่ไม่มี WARNING บันทึกอัตโนมัติ
    คนคุมจึงไม่มีจังหวะปฏิเสธรอบที่ทำท่าผิด (เกิดจริงกับ User6 extra day1 → ย้ายออกทั้งชุด)
    คนคุมต้องดูท่าตลอด 20 วิ ; ตรรกะการเก็บ/ระยะ/manifest ไม่เปลี่ยน
  - 26 ต.ค. 2026 (v6, Phase 5.6 แบบละเอียด, PROJECT_FACTS §5.1o): บันทึกบรรทัด Y (ผลทายของบอร์ดต่อเฟรม,
    เฟิร์มแวร์ MY_TOF_PRED_LINE=1) และ Q (เวลาต่อเฟรม, MY_TOF_FRAME_TIMING=1) ของเฟรมที่อยู่ในรอบ
    ลงไฟล์เสริม <ชื่อรอบ>_board.csv ข้างไฟล์ .npz — ใช้วัดความแม่นยำบนบอร์ดตรงๆ, เทียบบอร์ดกับ PC
    ทุกเฟรม และกระจายของเวลาต่อเฟรม
    ไม่เปลี่ยน: ไฟล์ .npz, manifest_collect.csv, ช่วงระยะ/การถือ/การทิ้งรอบ, ข้อความบนจอ, คำถาม y/r/s
    ไม่แสดงผลทายระหว่างเก็บ (A2) — จอบอกแค่จำนวนบรรทัด Y/Q ที่บันทึกได้ (เช็คว่าบันทึกครบ)
    st_dataset.load_dataset() อ่านเฉพาะ *.npz จึงไม่สนใจไฟล์ .csv นี้
    ถ้าเฟิร์มแวร์ไม่ส่ง Y/Q (สวิตช์ปิด) จะเตือนตอนตรวจบอร์ด แล้วเก็บต่อได้ตามปกติ (ไฟล์เสริมว่าง)
  - กฎ: ทิ้งได้ทั้ง "รอบ" ที่ระยะไม่ผ่าน แต่ห้ามคัดทิ้งทีละเฟรมทีหลัง — การกรองระดับเฟรม
    ใช้กฎ 100-400 มม. ของ ST ใน st_dataset.py เท่านั้น เพื่อให้เทียบกับ ST ได้ยุติธรรม

จุดที่อาจพลาด (failure modes):
  - ถ้าบอร์ดส่ง 16 ค่า/บรรทัด = ยังอยู่โหมด 4x4 สคริปต์จะหยุดพร้อมข้อความชัดเจน
  - ถ้าเฟรมหายจริงกลางรอบ (ไม่ใช่ช่วงต้น) จะไม่ถูกทิ้ง แต่จะนับใน missing_frames ให้เห็น
  - ถ้าเฟรมค้างมากกว่า LEAD_CHECK เฟรม (ไม่เคยเห็น) ชั้นที่ 2 จะไม่จับ — ดู stale_dropped
    และ missing_frames ใน manifest ทุกครั้ง ถ้า missing_frames > 0 ให้สงสัยไว้ก่อน
  - ทดสอบตรรกะทิ้งเฟรมค้างด้วยข้อมูลจำลองแล้ว ยังต้องยืนยันด้วย dry run บนบอร์ดจริงอีกรอบ
  - ระยะมือ = โซนที่ใกล้ที่สุด ถ้ามีของอื่นใกล้กว่ามือในมุมมอง (แขนเสื้อ ขอบโต๊ะ) ตัวเลขจะเป็นของสิ่งนั้น
  - ถ้าบรรทัดสดดูซ้อน/เลื่อนลง แปลว่า console แสดงสระไทยกว้างผิด → ใช้ Windows Terminal
    หรือขยายหน้าต่างให้กว้างกว่า 80 ตัวอักษร (ไม่กระทบข้อมูลที่บันทึก)
  - ถ้าเห็นตัวอักษรแปลกๆ เช่น "←[32m" แปลว่า console ไม่รองรับสี → ใส่ --no-color
  - ข้อความเตือน (WARNING) เป็นภาษาอังกฤษ — ความหมายภาษาไทยดูใน summarise()
  - ตรรกะรอ/ทิ้งรอบทดสอบด้วยข้อมูลจำลองแล้ว (9 ต.ค.) ยังต้อง dry run บนบอร์ดจริง 1 รอบ
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]
N_ZONES = 64
VALID_STATUS = (5, 9)
MIN_MM, MAX_MM = 100.0, 400.0          # ช่วงระยะที่ ST ใช้ (PROJECT_FACTS §5.1)
ZONE_HEAD = ["target_status", "valid", "signal_per_spad", "distance_mm"]
# รหัสตัวเลขของ ST ต่อท่า (คัดจาก HAND_POSTURE_DICT ใน st_dataset.py ซึ่งอ้าง
# utils/handposture_dictionnary.py ของ ST) เพื่อให้ st_dataset.load_dataset() อ่านข้อมูลเราได้
ST_LABEL = {"FlatHand": 20, "Like": 21, "Love": 24, "Dislike": 25,
            "BreakTime": 27, "CrossHands": 28, "Fist": 32}
STALE_FRAMES = 30       # เฟรมที่รอ S/G ไม่ครบเกินนี้ให้ทิ้ง
DRAIN_S = 0.5           # อ่านทิ้งกี่วินาทีหลังล้าง buffer (8x8 ≈ 66 ms/เฟรม → ราว 7 เฟรม)
STALE_GAP = 5           # เลขเฟรมกระโดดเกินนี้ในช่วงต้นรอบ = ข้อมูลก่อนหน้าคือเฟรมค้าง
LEAD_CHECK = 10         # ตรวจหาจุดกระโดดเฉพาะในกี่เฟรมแรก
HOLD_S = 3.0            # มือต้องอยู่ในช่วงระยะต่อเนื่องกี่วินาทีก่อนเริ่มเก็บ (Chin กำหนด 9 ต.ค.)
ABORT_FRAMES = 8        # ระหว่างเก็บ มือหลุดช่วงติดกันกี่เฟรมถึงทิ้งรอบ (8 ≈ 0.5 วิ ที่ 15 Hz)
ARM_TIMEOUT_S = 60.0    # รอมือเข้าช่วงนานสุดกี่วินาที แล้วถามว่าจะทำใหม่หรือข้าม
LINE_W = 76             # ความกว้างบรรทัดแสดงผลสด (เติมช่องว่างลบตัวอักษรเก่า, < 80 กันขึ้นบรรทัดใหม่)
BAR_W = 16              # ความยาวแถบความคืบหน้า (ตัวอักษร)
RULE_W = 66             # ความยาวเส้นคั่นหัวข้อ
YES_ANSWERS = ("y", "ั")   # "ั" = ปุ่ม Y ตอนแป้นเป็นภาษาไทย
REDO_ANSWERS = ("r", "พ")  # "พ" = ปุ่ม R ตอนแป้นเป็นภาษาไทย
SKIP_ANSWERS = ("s", "ห")  # "ห" = ปุ่ม S ตอนแป้นเป็นภาษาไทย
# คอลัมน์ของ manifest_collect.csv ตามลำดับที่ main() เขียน (ใช้ตรวจไฟล์เก่าที่หัวตารางไม่ตรง)
MANIFEST_COLS = ["saved_at", "user", "class", "take", "attempt", "n_frames", "missing_frames",
                 "stale_dropped", "incomplete", "frac_in_range", "median_nearest_mm",
                 "first_frame", "last_frame", "out_of_window_frames", "window_lo_mm",
                 "window_hi_mm", "hold_s", "abort_frames", "minutes_since_poweron", "fw_note",
                 "seconds", "warnings", "file"]


@dataclass
class Take:
    """ข้อมูลที่เก็บได้ 1 รอบ (1 ท่า)"""
    frame_nos: list[int] = field(default_factory=list)
    dist: list[np.ndarray] = field(default_factory=list)
    status: list[np.ndarray] = field(default_factory=list)
    sig: list[np.ndarray] = field(default_factory=list)
    incomplete: int = 0      # เฟรมที่ได้ F/S/G ไม่ครบแล้วถูกทิ้ง
    stale_dropped: int = 0   # เฟรมค้างจากก่อนเริ่มรอบที่ถูกทิ้ง
    wrong_mode: bool = False
    t_start: float = float("nan")   # Unix time ตอนเริ่มอ่าน
    t_end: float = float("nan")     # Unix time ตอนอ่านจบ


def parse_line(line: str) -> tuple[str, int, list[float]] | None:
    """แยกบรรทัด 'F,12,105,...' / 'S,12,5,...' / 'G,12,320,...' คืน None ถ้าไม่ใช่
    บรรทัด F/S/G ที่ถูกต้อง (บรรทัด R, และ T, จะถูกข้ามตรงนี้)"""
    parts = line.strip().split(",")
    if len(parts) < 3 or parts[0] not in ("F", "S", "G"):
        return None
    try:
        return parts[0], int(parts[1]), [float(v) for v in parts[2:]]
    except ValueError:
        return None


class BoardTap:
    """v6: ห่อ readline — ส่งทุกบรรทัดต่อให้ผู้เรียกเหมือนเดิม แต่จดบรรทัด Y/Q (ผลทาย/เวลาของบอร์ด)
    เก็บไว้ตามเลขเฟรมด้วย ฟังก์ชันอ่านเดิม (parse_line, frame_stream) ข้ามบรรทัด Y/Q อยู่แล้วจึงไม่กระทบ"""

    def __init__(self, readline: Callable[[], str]) -> None:
        self._readline = readline
        self.y: dict[int, list[str]] = {}
        self.q: dict[int, list[str]] = {}

    def __call__(self) -> str:
        raw = self._readline()
        if raw and raw[:2] in ("Y,", "Q,"):
            parts = raw.strip().split(",")
            try:
                frame_no = int(parts[1])
            except (IndexError, ValueError):
                return raw
            (self.y if parts[0] == "Y" else self.q)[frame_no] = parts[2:]
        return raw

    def clear(self) -> None:
        """ล้างของเก่า (เรียกหลัง drain ก่อนเริ่มแต่ละรอบ)"""
        self.y.clear()
        self.q.clear()


BOARD_COLS = (["frame", "board_class"] + [f"board_p{i}" for i in range(len(CLASSES))]
              + ["sensor_us", "preprocess_us", "inference_us", "decision_us", "total_us"])


def write_board_log(npz_path: Path, frame_nos: list[int], tap: BoardTap) -> tuple[Path, int, int]:
    """v6: เขียน <ชื่อรอบ>_board.csv 1 แถวต่อเฟรมของรอบ (ลำดับเดียวกับใน .npz)
    board_class: 0..6 = ท่าที่บอร์ดทาย, -1 = บอร์ดไม่ทาย (มือนอก 100–400 มม.), ว่าง = ไม่ได้รับบรรทัด Y
    คืน (path, จำนวนเฟรมที่มี Y, จำนวนเฟรมที่มี Q)"""
    path = npz_path.with_name(npz_path.stem + "_board.csv")
    n_y = n_q = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(BOARD_COLS)
        for f in frame_nos:
            y = tap.y.get(f)
            q = tap.q.get(f)
            cls = y[0] if y else ""
            probs = y[1:1 + len(CLASSES)] if y and len(y) >= 1 + len(CLASSES) else [""] * len(CLASSES)
            times = q[:5] if q and len(q) >= 5 else [""] * 5
            n_y += y is not None
            n_q += q is not None
            w.writerow([f, cls] + probs + times)
    return path, n_y, n_q


def drain(readline: Callable[[], str], seconds: float) -> int:
    """อ่านทิ้ง `seconds` วินาที เพื่อเทข้อมูลเก่าที่ reset_input_buffer() ล้างไม่หมด
    คืนจำนวนบรรทัดที่ทิ้ง (ไว้ดูประกอบ ไม่ได้ใช้ตัดสินอะไร)"""
    n = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if readline():
            n += 1
    return n


def read_take(readline: Callable[[], str], seconds: float,
              max_frames: int | None = None) -> Take:
    """อ่านเฟรมจาก serial ต่อเนื่อง `seconds` วินาที (หรือจนครบ max_frames)
    ประกอบ F/S/G ของเลขเฟรมเดียวกันให้ครบก่อนนับเป็น 1 เฟรม"""
    take = Take(t_start=time.time())
    pending: dict[int, dict[str, list[float]]] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if max_frames is not None and len(take.frame_nos) >= max_frames:
            break
        raw = readline()
        if not raw:
            continue
        parsed = parse_line(raw)
        if parsed is None:
            continue
        kind, frame_no, values = parsed
        if len(values) == 16:
            take.wrong_mode = True      # บอร์ดยังอยู่โหมด 4x4
            continue
        if len(values) != N_ZONES:
            continue
        slot = pending.setdefault(frame_no, {})
        slot[kind] = values
        if all(k in slot for k in ("F", "S", "G")):
            take.frame_nos.append(frame_no)
            take.dist.append(np.array(slot["F"]))
            take.status.append(np.array(slot["S"]))
            take.sig.append(np.array(slot["G"]))
            del pending[frame_no]
        for old in [k for k in pending if frame_no - k > STALE_FRAMES]:
            del pending[old]
            take.incomplete += 1
    take.t_end = time.time()
    return take


def drop_stale_lead(take: Take, gap: int = STALE_GAP, lead: int = LEAD_CHECK) -> int:
    """ทิ้งเฟรมค้างที่หัวรอบ: ถ้าใน `lead` เฟรมแรกมีเลขเฟรมกระโดดเกิน `gap`
    ทุกเฟรมก่อนจุดกระโดด (จุดสุดท้ายในช่วงนั้น) ถือเป็นข้อมูลก่อนเริ่มรอบ แล้วทิ้ง
    เฟรมที่หายกลางรอบ (หลังช่วง lead) ไม่แตะ เพื่อให้ยังนับ missing_frames ได้ตามจริง
    คืนจำนวนเฟรมที่ทิ้ง และบันทึกลง take.stale_dropped ด้วย"""
    if len(take.frame_nos) < 2:
        return 0
    nos = np.array(take.frame_nos[: lead + 1])
    jumps = np.abs(np.diff(nos))                  # abs กันกรณีเลขเฟรมย้อนกลับ
    big = np.nonzero(jumps > gap)[0]
    if big.size == 0:
        return 0
    cut = int(big[-1]) + 1                        # ทิ้งเฟรมดัชนี 0..cut-1
    for lst in (take.frame_nos, take.dist, take.status, take.sig):
        del lst[:cut]
    take.stale_dropped += cut
    return cut


def nearest_mm(dist: np.ndarray, status: np.ndarray) -> float:
    """ระยะของโซนที่ใกล้ที่สุดในเฟรม (นับเฉพาะโซน status 5/9) = ระยะมือ
    คืน nan ถ้าไม่มีโซนที่ใช้ได้เลย"""
    valid = np.isin(status, VALID_STATUS)
    if not valid.any():
        return float("nan")
    return float(dist[valid].min())


def in_window(mm: float, lo: float, hi: float) -> bool:
    """ระยะอยู่ในช่วงที่ต้องการไหม (nan = ไม่เจอมือ = ไม่อยู่)"""
    return (not np.isnan(mm)) and lo <= mm <= hi


def frame_stream(readline: Callable[[], str]):
    """generator: ประกอบ F/S/G ของเลขเฟรมเดียวกันแล้ว yield (frame_no, dist, status, sig)
    yield None เมื่ออ่านไม่ได้อะไร (serial timeout) เพื่อให้ผู้เรียกเช็คเวลาได้ต่อ
    ข้ามบรรทัดที่ไม่ใช่ 64 ค่า (โหมด 4x4 ถูกจับไปแล้วตอนตรวจก่อนเก็บ)"""
    pending: dict[int, dict[str, list[float]]] = {}
    while True:
        raw = readline()
        if not raw:
            yield None
            continue
        parsed = parse_line(raw)
        if parsed is None:
            continue
        kind, frame_no, values = parsed
        if len(values) != N_ZONES:
            continue
        slot = pending.setdefault(frame_no, {})
        slot[kind] = values
        if all(k in slot for k in ("F", "S", "G")):
            del pending[frame_no]
            yield frame_no, np.array(slot["F"]), np.array(slot["S"]), np.array(slot["G"])
        for old in [k for k in pending if frame_no - k > STALE_FRAMES]:
            del pending[old]


# ---------------------------------------------------------------- การแสดงผล
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
COLOR_CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
_use_color = False      # ตั้งใน main() หลังเช็คว่า console รองรับสีไหม


def enable_color(want: bool) -> bool:
    """เปิดสี ANSI ถ้า console รองรับ (Windows 10+ ต้องเปิดโหมด VT ก่อน) ไม่ได้ก็ไม่ใช้สี"""
    if not want or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)               # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):   # ENABLE_VT_PROCESSING
                return False
        except (AttributeError, OSError):
            return False
    return True


def paint(text: str, color: str) -> str:
    """ใส่สีให้ข้อความ (ถ้าปิดสีจะคืนข้อความเดิม)"""
    return f"\x1b[{COLOR_CODES[color]}m{text}\x1b[0m" if _use_color else text


def visual_width(text: str) -> int:
    """ความกว้างที่เห็นบนจอ: ไม่นับรหัสสี และสระบน/ล่าง วรรณยุกต์ไทย (หมวด Mn) ไม่กินช่อง
    ใช้แทน len() ตอนเติมช่องว่าง ไม่งั้นข้อความที่มีวรรณยุกต์มากจะเหลือเศษตัวอักษรเก่าท้ายบรรทัด"""
    plain = ANSI_RE.sub("", text)
    return sum(0 if unicodedata.category(ch) in ("Mn", "Me", "Cf") else 1 for ch in plain)


def pad(text: str, width: int) -> str:
    """เติมช่องว่างด้านขวาให้กว้างเท่า width ตามความกว้างที่เห็นจริง (ใช้จัดคอลัมน์ภาษาไทย)"""
    return text + " " * max(0, width - visual_width(text))


def show_live(text: str) -> None:
    """แสดงข้อความบรรทัดเดียวที่อัปเดตทับที่เดิม (\\r) ไม่ไหลลงเป็นแถว"""
    print("\r" + pad(text, LINE_W), end="", flush=True)


def beep(enabled: bool) -> None:
    """เสียงบี๊ปของ console บอกว่าเริ่ม/จบการเก็บ จะได้ไม่ต้องละสายตาจากมือ"""
    if enabled:
        print("\a", end="", flush=True)


def progress_bar(frac: float, width: int = BAR_W) -> str:
    """แถบความคืบหน้า █ (ทำแล้ว) ░ (ที่เหลือ)"""
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


def distance_tag(mm: float, lo: float, hi: float) -> str:
    """ป้ายสถานะระยะ (บอกแค่ระยะ ไม่บอกผลทายของโมเดล — A6 ป้ายมาจากเจตนา)"""
    if np.isnan(mm):
        return paint("[NO HAND]  ", "red")
    if mm > hi:
        return paint("[TOO FAR]  ", "yellow")
    if mm < lo:
        return paint("[TOO CLOSE]", "yellow")
    return paint("[IN RANGE] ", "green")


def rule(title: str = "", char: str = "─") -> str:
    """เส้นคั่นหัวข้อ เช่น ──── Fist ─────"""
    if not title:
        return char * RULE_W
    head = f"{char * 4} {title} "
    return head + char * max(4, RULE_W - visual_width(head))


def kv(label: str, value: str, width: int = 19, prefix: str = "  │ ") -> str:
    """บรรทัด "หัวข้อ   ค่า" จัดคอลัมน์ให้ตรงกันแม้มีสระไทย"""
    return prefix + pad(label, width) + value


def arm_and_record(stream, seconds: float, lo: float, hi: float, hold_s: float,
                   abort_frames: int, arm_timeout_s: float,
                   display: Callable[[str], None] = show_live,
                   sound: bool = True) -> tuple[Take | None, str, int]:
    """ขั้น A (รอมือ): แสดงระยะสด รอจนมืออยู่ในช่วง [lo, hi] ต่อเนื่อง hold_s วินาที
         ถ้าหลุดช่วงระหว่างนับ → นับใหม่จาก 0 ; รอเกิน arm_timeout_s → คืน 'timeout'
       ขั้น B (เก็บ): เก็บต่อเนื่อง `seconds` วินาทีจาก stream เดิม (ไม่ล้าง buffer → ไม่มีช่องว่าง)
         มือหลุดช่วงติดกัน >= abort_frames เฟรม → ทิ้งทั้งรอบ คืน 'aborted'
         หลุดสั้นกว่านั้น → ไม่ทิ้งเฟรม (ไม่คัดทีละเฟรม) แต่นับรวมไว้ใน out_frames
       คืน (take หรือ None, สถานะ 'ok'/'aborted'/'timeout', out_frames)"""
    # ---- ขั้น A: รอมือเข้าช่วงและค้างครบ hold_s ----
    t0 = time.monotonic()
    hold_start: float | None = None
    for item in stream:
        now = time.monotonic()
        if now - t0 > arm_timeout_s:
            print()
            return None, "timeout", 0
        if item is None:
            display("  Waiting for data from the board ...")
            continue
        _, d, s, _ = item
        mm = nearest_mm(d, s)
        if in_window(mm, lo, hi):
            hold_start = hold_start if hold_start is not None else now
        else:
            hold_start = None
        held = (now - hold_start) if hold_start is not None else 0.0
        mm_txt = "----" if np.isnan(mm) else f"{mm:4.0f}"
        frac = held / hold_s if hold_s > 0 else 1.0
        display(f"  Distance {paint(mm_txt, 'bold')} mm  {distance_tag(mm, lo, hi)}  "
                f"Hold {progress_bar(frac)} {min(held, hold_s):3.1f}/{hold_s:.1f} s")
        if held >= hold_s:
            break
    print()
    beep(sound)
    # ---- ขั้น B: เก็บจริง ----
    take = Take(t_start=time.time())
    out_run = 0
    out_frames = 0
    t1 = time.monotonic()
    for item in stream:
        now = time.monotonic()
        if now - t1 >= seconds:
            break
        if item is None:
            continue
        frame_no, d, s, g = item
        take.frame_nos.append(frame_no)
        take.dist.append(d)
        take.status.append(s)
        take.sig.append(g)
        mm = nearest_mm(d, s)
        if in_window(mm, lo, hi):
            out_run = 0
        else:
            out_run += 1
            out_frames += 1
            if out_run >= abort_frames:
                print()
                beep(sound)
                return None, "aborted", out_frames
        mm_txt = "----" if np.isnan(mm) else f"{mm:4.0f}"
        el = now - t1
        display(f"  {paint('● REC', 'red')}  Distance {paint(mm_txt, 'bold')} mm  "
                f"{distance_tag(mm, lo, hi)} {progress_bar(el / seconds)} "
                f"{el:4.1f}/{seconds:.0f} s  Out {out_frames}")
    take.t_end = time.time()
    print()
    beep(sound)
    return take, "ok", out_frames


def summarise(take: Take, target_mm: float, tol_mm: float) -> tuple[dict, list[str]]:
    """สรุปคุณภาพของรอบนี้ + รายการคำเตือน (ว่าง = ผ่าน)"""
    n = len(take.frame_nos)
    warns: list[str] = []
    if take.wrong_mode:
        warns.append("Board sends 16 values/line = 4x4 mode — set MY_TOF_USE_4X4=0 and re-flash")
    if n == 0:
        return {"n_frames": 0}, warns + ["No frames received — check cable, COM port and firmware data mode"]

    status = np.stack(take.status)            # (N, 64)
    dist = np.stack(take.dist)
    valid = np.isin(status, VALID_STATUS)     # (N, 64)
    frac_any_valid = float(valid.any(axis=1).mean())
    if frac_any_valid < 0.5:
        warns.append("Almost no zones with status 5 or 9 — firmware may use BSP remap "
                     "(FAST_READ=0?) or nothing is in range")
    nearest = np.where(valid, dist, 4000.0).min(axis=1)     # ระยะของจุดที่ใกล้ที่สุดในแต่ละเฟรม
    in_range = (nearest >= MIN_MM) & (nearest <= MAX_MM)
    frac_in_range = float(in_range.mean())
    median_mm = float(np.median(nearest[in_range])) if in_range.any() else float("nan")
    if frac_in_range < 0.8:
        warns.append(f"Hand within {MIN_MM:.0f}-{MAX_MM:.0f} mm in only {frac_in_range*100:.0f}% of frames")
    if in_range.any() and abs(median_mm - target_mm) > tol_mm:
        warns.append(f"Median distance {median_mm:.0f} mm is outside {target_mm:.0f} ± {tol_mm:.0f} mm")

    diffs = np.diff(np.array(take.frame_nos))
    missing = int((diffs[diffs > 1] - 1).sum()) if diffs.size else 0
    if missing > 0:
        warns.append(f"{missing} frames missing mid-take (real frame loss) — redo this take")
    info = {"n_frames": n, "missing_frames": missing, "stale_dropped": take.stale_dropped,
            "incomplete": take.incomplete,
            "frac_in_range": round(frac_in_range, 3), "median_nearest_mm": round(median_mm, 1),
            "first_frame": take.frame_nos[0], "last_frame": take.frame_nos[-1]}
    return info, warns


def save_take(take: Take, out_dir: Path, class_name: str, user: int, k: int) -> Path:
    """บันทึกเป็น .npz ตามโครงสร้าง <out>/<Class>/log__...__User<N>__take<k>/npz/*.npz"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"log__{stamp}__User{user}__take{k}"
    folder = out_dir / class_name / log_name / "npz"
    folder.mkdir(parents=True, exist_ok=True)

    status = np.stack(take.status).T.astype(np.float64)     # (64, N)
    dist = np.stack(take.dist).T.astype(np.float64)
    sig = np.stack(take.sig).T.astype(np.float64)
    valid = np.isin(status, VALID_STATUS).astype(np.float64)
    zone_data = np.stack([status, valid, sig, dist], axis=0)  # (4, 64, N)
    n = zone_data.shape[-1]
    glob = np.full((1, n), float(ST_LABEL[class_name]))
    path = folder / f"{log_name}.npz"
    np.savez(path, start_tstmp=np.float64(take.t_start), end_tstmp=np.float64(take.t_end),
             zone_data=zone_data, glob_data=glob,
             zone_head=np.array(ZONE_HEAD), glob_head=np.array([".GestureGT"]))
    return path


def append_manifest(path: Path, row: dict) -> None:
    """เพิ่ม 1 แถวต่อท้าย manifest (สร้างหัวตารางถ้ายังไม่มีไฟล์)
    ถ้าไฟล์เดิมมีหัวตารางไม่ตรง (เช่นสร้างจากสคริปต์รุ่นก่อน) ให้หยุด กันคอลัมน์เลื่อน"""
    new = not path.exists()
    if not new:
        with path.open(newline="", encoding="utf-8") as fh:
            old_header = next(csv.reader(fh), [])
        if old_header != list(row.keys()):
            raise SystemExit(f"Existing manifest {path} has different columns — use a new --out-dir "
                             f"(the take just saved is in {path.parent})")
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def is_yes(answer: str) -> bool:
    """รับ y/Y และ "ั" (ปุ่ม Y ตอนแป้นเป็นภาษาไทย) ดูแค่ตัวอักษรแรก"""
    a = answer.strip().lower()
    return bool(a) and a[0] in YES_ANSWERS


def parse_action(answer: str, allow_save: bool = True) -> str | None:
    """แปลงคำตอบเป็น 'save' / 'redo' / 'skip' — ตัวอักษรไทยบนปุ่มเดียวกันก็รับ
    (y = "ั", r = "พ", s = "ห" บนแป้นเกษมณี) คืน None ถ้าไม่รู้จัก เพื่อให้ถามซ้ำ
    แทนที่จะข้ามไปเงียบๆ (กันพลาดแบบ dry run 8 ต.ค. ที่รอบหายเพราะพิมพ์ผิดแป้น)"""
    a = answer.strip().lower()
    if not a:
        return None
    ch = a[0]
    if allow_save and ch in YES_ANSWERS:
        return "save"
    if ch in REDO_ANSWERS:
        return "redo"
    if ch in SKIP_ANSWERS:
        return "skip"
    return None


def ask_action(prompt: str, allow_save: bool = True) -> str:
    """ถามจนกว่าจะได้คำตอบที่รู้จัก"""
    while True:
        action = parse_action(input(prompt), allow_save)
        if action is not None:
            return action
        print("  Unknown answer — type y, r or s (check the keyboard is set to English)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="เช่น COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--user", type=int, required=True, help="หมายเลขผู้ทดลอง (ควรเริ่มที่ 5 เพราะ 1-4 คือของ ST)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/phase4"))
    ap.add_argument("--takes", type=int, default=2, help="จำนวนรอบต่อท่า")
    ap.add_argument("--seconds", type=float, default=20.0, help="วินาทีต่อรอบ")
    ap.add_argument("--target-mm", type=float, default=270.0)
    ap.add_argument("--tol-mm", type=float, default=50.0)
    ap.add_argument("--minutes-since-poweron", type=float, default=float("nan"))
    ap.add_argument("--fw-note", default="", help="บันทึกเฟิร์มแวร์ เช่น commit hash")
    ap.add_argument("--hold-s", type=float, default=HOLD_S,
                    help="มือต้องค้างในช่วงกี่วินาทีก่อนเริ่มเก็บ")
    ap.add_argument("--abort-frames", type=int, default=ABORT_FRAMES,
                    help="ระหว่างเก็บ หลุดช่วงติดกันกี่เฟรมถึงทิ้งรอบ (1 = เข้มสุด ไม่แนะนำ)")
    ap.add_argument("--arm-timeout-s", type=float, default=ARM_TIMEOUT_S,
                    help="รอมือเข้าช่วงนานสุดกี่วินาที")
    ap.add_argument("--no-beep", action="store_true", help="ปิดเสียงบี๊ปตอนเริ่ม/จบการเก็บ")
    ap.add_argument("--no-color", action="store_true", help="ปิดสี (ถ้า console แสดงรหัสแปลกๆ)")
    args = ap.parse_args()
    if args.abort_frames < 1 or args.hold_s < 0 or args.tol_mm <= 0:
        raise SystemExit("--abort-frames must be >= 1, --hold-s >= 0, --tol-mm > 0")
    lo, hi = args.target_mm - args.tol_mm, args.target_mm + args.tol_mm

    global _use_color
    _use_color = enable_color(not args.no_color)
    if args.user <= 4:
        print(paint("[WARNING]", "yellow") + " Users 1-4 belong to the ST dataset — use 5 or higher")
    try:
        import serial   # นำเข้าตรงนี้เพื่อให้ทดสอบส่วนอื่นได้โดยไม่ต้องมี pyserial
    except ImportError:
        raise SystemExit("pyserial is required: pip install pyserial")

    ser = serial.Serial(args.port, args.baud, timeout=1)
    # v6: ห่อด้วย BoardTap เพื่อจดบรรทัด Y/Q ของบอร์ด (ส่วนอื่นอ่านเหมือนเดิมทุกอย่าง)
    readline = BoardTap(lambda: ser.readline().decode("utf-8", errors="ignore"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest_collect.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            old_header = next(csv.reader(fh), [])
        if old_header != MANIFEST_COLS:
            ser.close()
            raise SystemExit(f"{manifest} was made by an older script version (columns differ) — use a new --out-dir")

    # ---- กรอบข้อมูลการเก็บ (แคปใส่รายงานได้) ----
    print()
    print(paint(rule(char="═"), "cyan"))
    print(paint("  Hand Posture Data Collection  ·  VL53L8CA 8×8  ·  NUCLEO-F411RE", "bold"))
    print(paint(rule(char="═"), "cyan"))
    print(kv("Subject", f"User {args.user}", prefix="  "))
    print(kv("Postures", ", ".join(CLASSES), prefix="  "))
    print(kv("Distance window", f"{lo:.0f}–{hi:.0f} mm  ({args.target_mm:.0f} ± {args.tol_mm:.0f})", prefix="  "))
    print(kv("Hold before rec.", f"{args.hold_s:.1f} s continuous (resets if hand leaves window)", prefix="  "))
    print(kv("Recording time", f"{args.seconds:g} s per take", prefix="  "))
    print(kv("Discard rule", f"{args.abort_frames} consecutive out-of-window frames (~{args.abort_frames / 15:.1f} s)",
             prefix="  "))
    print(paint(rule(char="═"), "cyan"))

    # ---- ตรวจบอร์ด: ยื่นมือก่อน แล้วกด Enter (dry run 3: ยื่นไม่ทันเมื่อเริ่มอ่านทันที) ----
    input("\n  Board check — hold your hand in front of the sensor, then press Enter ")
    ser.reset_input_buffer()
    drain(readline, DRAIN_S)
    pre = read_take(readline, 3.0)
    drop_stale_lead(pre)
    info, warns = summarise(pre, args.target_mm, args.tol_mm)
    hard = [w for w in warns if "4x4" in w or "No frames" in w or "status" in w]
    n_pre = info.get("n_frames", 0)
    dur = pre.t_end - pre.t_start
    hz = n_pre / dur if dur > 0 else float("nan")
    verdict = paint("[FAIL]", "red") if hard else paint("[PASS]", "green")
    print(f"  Board check   {n_pre} frames / {dur:.1f} s = {hz:.1f} Hz   {verdict}")
    for w in warns:
        print("  " + paint("[WARNING]", "yellow") + f" {w}")
    # v6: เช็คว่าเฟิร์มแวร์ส่งบรรทัด Y/Q มาไหม (ไม่ใช่เงื่อนไขหยุด — แค่เตือนว่าไฟล์เสริมจะว่าง)
    n_y_pre = sum(f in readline.y for f in pre.frame_nos)
    n_q_pre = sum(f in readline.q for f in pre.frame_nos)
    print(f"  Board log     Y lines {n_y_pre}/{n_pre} frames, Q lines {n_q_pre}/{n_pre} frames")
    if n_pre and n_y_pre == 0:
        print("  " + paint("[WARNING]", "yellow") + " no Y lines — firmware MY_TOF_PRED_LINE is 0? "
              "the _board.csv files will be empty")
    if hard:
        ser.close()
        raise SystemExit("Stopped: fix the warning above before collecting data")

    tally = {c: {"saved": 0, "aborted": 0, "skipped": 0} for c in CLASSES}
    try:
        for ci, cls in enumerate(CLASSES, start=1):
            for k in range(1, args.takes + 1):
                attempt = 0
                while True:                         # วนจนกว่าจะเก็บได้ หรือผู้คุมเลือกข้าม
                    attempt += 1
                    title = f"Posture {ci}/{len(CLASSES)}  {cls}  ·  Take {k}/{args.takes}"
                    if attempt > 1:
                        title += f"  ·  Attempt {attempt}"
                    print("\n" + paint(rule(title), "cyan"))
                    print(paint(f"  Form the posture → move into {lo:.0f}–{hi:.0f} mm → hold {args.hold_s:g} s "
                                f"→ auto-record {args.seconds:g} s", "dim"))
                    ans = input("  Press Enter to start  (s = skip this take) ")
                    if parse_action(ans, allow_save=False) == "skip":
                        print("  " + paint("[SKIPPED]", "yellow") + " not saved")
                        tally[cls]["skipped"] += 1
                        break
                    ser.reset_input_buffer()
                    drain(readline, DRAIN_S)        # เทข้อมูลเก่าที่ค้างระหว่างรอ Enter ทิ้ง
                    readline.clear()                # v6: ล้างบรรทัด Y/Q เก่า
                    take, state, out_frames = arm_and_record(
                        frame_stream(readline), args.seconds, lo, hi, args.hold_s,
                        args.abort_frames, args.arm_timeout_s, sound=not args.no_beep)
                    if state == "aborted":
                        print("  " + paint("[DISCARDED]", "red") +
                              f" hand left the window for {args.abort_frames} consecutive frames "
                              f"(~{args.abort_frames / 15:.1f} s) — not saved, retrying")
                        tally[cls]["aborted"] += 1
                        continue
                    if state == "timeout":
                        print("  " + paint("[TIMEOUT]", "yellow") +
                              f" hand did not hold in the window within {args.arm_timeout_s:.0f} s")
                        action = ask_action("  r = retry, s = skip : ", allow_save=False)
                        if action == "skip":
                            print("  " + paint("[SKIPPED]", "yellow") + " not saved")
                            tally[cls]["skipped"] += 1
                            break
                        continue
                    info, warns = summarise(take, args.target_mm, args.tol_mm)
                    n = info.get("n_frames", 0)
                    dur = take.t_end - take.t_start
                    print(paint("  Result", "bold"))
                    print(kv("Frames recorded", f"{n}  ({n / dur:.1f} Hz)" if dur > 0 else f"{n}"))
                    print(kv("Missing frames", f"{info.get('missing_frames', 0)}"))
                    print(kv("Out-of-window", f"{out_frames} frames (brief)"))
                    print(kv("Median distance", f"{info.get('median_nearest_mm', float('nan'))} mm"))
                    for w in warns:
                        print("  │ " + paint("[WARNING]", "yellow") + f" {w}")
                    if n == 0:
                        action = ask_action("  └ No frames — r = retry, s = skip : ", allow_save=False)
                    elif warns:
                        action = ask_action("  └ Warning — y = save anyway, r = retry, s = skip : ")
                    else:
                        # v5: ถามคนคุมทุกรอบ — โปรแกรมเช็คได้แค่ระยะ ไม่รู้ว่าท่าถูกไหม (§9 #53)
                        action = ask_action("  └ Posture correct the whole time? "
                                            "y = save, r = redo, s = skip : ")
                    if action == "redo":
                        continue
                    if action == "skip":
                        print("  └ " + paint("[SKIPPED]", "yellow") + " not saved")
                        tally[cls]["skipped"] += 1
                        break
                    path = save_take(take, args.out_dir, cls, args.user, k)
                    blog, n_y, n_q = write_board_log(path, take.frame_nos, readline)   # v6
                    append_manifest(manifest, {
                        "saved_at": datetime.now().isoformat(timespec="seconds"), "user": args.user,
                        "class": cls, "take": k, "attempt": attempt, **info,
                        "out_of_window_frames": out_frames, "window_lo_mm": lo, "window_hi_mm": hi,
                        "hold_s": args.hold_s, "abort_frames": args.abort_frames,
                        "minutes_since_poweron": args.minutes_since_poweron, "fw_note": args.fw_note,
                        "seconds": args.seconds, "warnings": " | ".join(warns), "file": str(path)})
                    tally[cls]["saved"] += 1
                    try:
                        shown = path.relative_to(args.out_dir)
                    except ValueError:
                        shown = path
                    print("  └ " + paint("[SAVED]", "green") + f" {shown.name}")
                    print(paint(f"    board log: Y {n_y}/{n}, Q {n_q}/{n} frames → {blog.name}", "dim"))
                    break
    except KeyboardInterrupt:
        print("\n\n  " + paint("[STOPPED]", "yellow") + " takes already saved are kept")
    finally:
        ser.close()
        print_tally(tally, args.takes)


def print_tally(tally: dict[str, dict[str, int]], takes: int) -> None:
    """ตารางสรุปตอนจบ: แต่ละท่าบันทึกได้กี่รอบ ทิ้งกี่ครั้ง ข้ามกี่รอบ"""
    print()
    print(paint(rule("Collection Summary", char="═"), "cyan"))
    print("  " + pad("Posture", 14) + pad("Saved", 10) + pad("Discarded", 12) + "Skipped")
    print("  " + "─" * 43)
    for cls, t in tally.items():
        saved = f"{t['saved']}/{takes}"
        saved = paint(saved, "green") if t["saved"] == takes else saved
        print("  " + pad(cls, 14) + pad(saved, 10) + pad(str(t["aborted"]), 12) + str(t["skipped"]))
    total = sum(t["saved"] for t in tally.values())
    print("  " + "─" * 43)
    print("  " + pad("Total", 14) + f"{total}/{takes * len(tally)}")
    print(paint(rule(char="═"), "cyan"))


if __name__ == "__main__":
    main()
