"""recentre.py — 5.5c: ย้ายมือให้อยู่กลางภาพ 8×8 ก่อนส่งเข้าโมเดล (PROJECT_FACTS §5.1w)

ใช้ตัวเดียวกันทุกที่ (ฝึก, ทดสอบ, PC re-check) และเฟิร์มแวร์ต้องทำ "เลขจำนวนเต็มเหมือนกันเป๊ะ"
  1. โซนมือ = โซนที่ระยะ (หลัง normalise) ไม่ใช่ค่าว่าง  → ในข้อมูลที่เตรียมแล้ว ค่าว่าง = (4000-295)/196 ≈ 18.9
  2. cnt = จำนวนโซนมือ, sum_r = ผลรวมเลขแถว, sum_c = ผลรวมเลขคอลัมน์ (ทั้งหมดเป็นจำนวนเต็ม)
  3. เลื่อน dr = floor((4*cnt - sum_r) / cnt), dc = floor((4*cnt - sum_c) / cnt)
       = floor(3.5 - centroid + 0.5) → จุดศูนย์กลางหลังเลื่อนอยู่ในช่วง (3.0, 4.0]
     คิดด้วยจำนวนเต็มล้วน (floor division) → Python กับ C ได้ค่าเดียวกันทุกเฟรม ไม่มีปัญหาปัดเศษทศนิยม
     (ใน C ต้องเขียน floor division เอง เพราะ / ของ C ปัดเข้าหาศูนย์เมื่อเป็นลบ)
  4. เลื่อนทั้ง 2 ช่อง (distance, signal); ช่องที่ว่างจากการเลื่อน = ค่าว่าง (ระยะ 4000, signal 0 หลัง normalise)
     ช่องที่เลื่อนหลุดตาราง = ทิ้ง ; เฟรมที่ไม่มีโซนมือ = ไม่เลื่อน
"""
from __future__ import annotations

import numpy as np

DIST_MEAN, DIST_STD = 295.0, 196.0
SIG_MEAN, SIG_STD = 281.0, 452.0
EMPTY_DIST = (4000.0 - DIST_MEAN) / DIST_STD      # ≈ 18.903 ค่าระยะของโซนว่างหลัง normalise
EMPTY_SIG = (0.0 - SIG_MEAN) / SIG_STD             # ≈ -0.622
HAND_THRESH = (2000.0 - DIST_MEAN) / DIST_STD      # ระยะ < 2000 มม. = โซนมือ (มือจริง ≤ 520 มม.)


def shifts(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """คืน (dr, dc, cnt) ต่อเฟรม — X: (n, 8, 8, 2) ที่เตรียมแล้ว"""
    hand = X[..., 0] < HAND_THRESH
    cnt = hand.sum(axis=(1, 2)).astype(np.int64)
    r = np.arange(8)[None, :, None]
    c = np.arange(8)[None, None, :]
    sum_r = (hand * r).sum(axis=(1, 2)).astype(np.int64)
    sum_c = (hand * c).sum(axis=(1, 2)).astype(np.int64)
    safe = np.maximum(cnt, 1)
    dr = np.where(cnt > 0, (4 * cnt - sum_r) // safe, 0)     # // ของ numpy = floor division
    dc = np.where(cnt > 0, (4 * cnt - sum_c) // safe, 0)
    return dr.astype(int), dc.astype(int), cnt


def shift_frames(X: np.ndarray, dr: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """เลื่อนแต่ละเฟรมตาม dr, dc (จำนวนเต็ม) เติมช่องว่างด้วยค่าว่าง"""
    out = np.empty_like(X)
    out[..., 0] = EMPTY_DIST
    out[..., 1] = EMPTY_SIG
    for a, b in set(zip(dr.tolist(), dc.tolist())):
        idx = np.nonzero((dr == a) & (dc == b))[0]
        if abs(a) >= 8 or abs(b) >= 8:
            continue
        rs, rd = slice(max(a, 0), 8 + min(a, 0)), slice(max(-a, 0), 8 + min(-a, 0))
        cs, cd = slice(max(b, 0), 8 + min(b, 0)), slice(max(-b, 0), 8 + min(-b, 0))
        out[idx, rs, cs, :] = X[idx][:, rd, cd, :]
    return out


def recentre(X: np.ndarray) -> np.ndarray:
    """ย้ายมือทุกเฟรมให้จุดศูนย์กลางอยู่ใกล้กลางภาพ (3.0, 4.0] ทั้งแถวและคอลัมน์"""
    if len(X) == 0:
        return X.copy()
    dr, dc, _ = shifts(X)
    return shift_frames(X, dr, dc)
