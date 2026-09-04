"""
st_dataset.py
=============
ตัวโหลดชุดข้อมูล ST_VL53L8CX_handposture_dataset ที่เราเขียนเอง

เหตุผลที่เขียนเอง (ไม่ใช้ data_loader.py ของ ST):
  1. ของ ST สับข้อมูลทุกเฟรมปนกันแล้วตัด 80/20 (data_loader.py:184-191)
     -> ภาพของคนเดียวกันไปอยู่ทั้ง train และ test = คะแนนสูงเกินจริง
  2. ของ ST ไม่เก็บ "รหัสคน" (User1..User4) ไว้เลย
     -> ทำ subject-independent split ไม่ได้
  ตัวนี้ดึงรหัสคนจากชื่อโฟลเดอร์เก็บไว้ด้วย จึงแบ่งแบบแยกคนได้

ขั้นตอน preprocessing ทำตาม ST ทุกประการ (data_loader.py:100-160)
เพื่อให้เทียบผลกับ ST ได้อย่างเป็นธรรม -- เปลี่ยนแค่ "วิธีแบ่งข้อมูล" อย่างเดียว

โครงสร้างชื่อโฟลเดอร์ในชุดข้อมูล:
    Fist/log__Fist__User2__1__VL53L8__handposture_api__8x8__20230504_105045
         ^ท่า      ^คน  ^ครั้งที่                              ^วันเวลา

วิธีรัน:
    python st_dataset.py --data-dir ./ST_VL53L8CX_handposture_dataset
    python st_dataset.py --data-dir ./ST_VL53L8CX_handposture_dataset --drop-none
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── ค่าคงที่จากฝั่ง ST — ห้ามเปลี่ยนถ้ายังต้องการเทียบผลกับเขา ──────────────

# สถานะที่ถือว่าโซนนั้นวัดได้จริง (data_loader.py:118)
# 5 = มั่นใจ 100% · 9 = มั่นใจ 50%
# หมายเหตุสำคัญ: นี่คือค่า "ดิบ" จากเซ็นเซอร์
# บอร์ดของเราผ่าน BSP ซึ่งแปลง 5/9 -> 0 แล้ว (PROJECT_FACTS TRAP #3)
# ตอนเอาข้อมูลบอร์ดเรามาเข้าโมเดลนี้ ต้องแปลงกลับก่อน
VALID_STATUS: tuple[int, ...] = (5, 9)

DEFAULT_DISTANCE: float = 4000.0  # ระยะที่ใส่แทนโซนที่ใช้ไม่ได้ (data_loader.py:119)
DEFAULT_SIGNAL: float = 0.0       # ความแรงแสงที่ใส่แทนโซนที่ใช้ไม่ได้

# ค่าที่ใช้ทำ normalization (data_loader.py:156-157)
DIST_MEAN, DIST_STD = 295.0, 196.0
SIG_MEAN, SIG_STD = 281.0, 452.0

# ค่ากรองเริ่มต้นจาก training_config.yaml (หน่วยเป็น มม.)
MAX_DISTANCE_MM: float = 400.0       # มือไกลกว่านี้ = ทิ้งทั้งเฟรม
MIN_DISTANCE_MM: float = 100.0       # มือใกล้กว่านี้ = ทิ้งทั้งเฟรม
BACKGROUND_DISTANCE_MM: float = 120.0  # ลึกกว่าจุดที่ใกล้ที่สุดเกินค่านี้ = ถือเป็นฉากหลัง

N_ZONES: int = 64  # 8x8

# ลำดับช่องข้อมูลใน zone_data ของไฟล์ .npz
CH_TARGET_STATUS: int = 0
CH_VALID: int = 1
CH_SIGNAL: int = 2
CH_DISTANCE: int = 3

# แผนที่ชื่อท่า -> เลข label (utils/handposture_dictionnary.py)
# เก็บเฉพาะ 8 คลาสที่มีอยู่จริงในชุดข้อมูลนี้
HAND_POSTURE_DICT: dict[str, int] = {
    "None": 0,
    "FlatHand": 20,
    "Like": 21,
    "Love": 24,
    "Dislike": 25,
    "BreakTime": 27,
    "CrossHands": 28,
    "Fist": 32,
}

# ดึงรหัสคนจากชื่อโฟลเดอร์ เช่น "__User2__" -> "User2"
_USER_PATTERN = re.compile(r"__User(\d+)__")


@dataclass
class HandPostureData:
    """ชุดข้อมูลที่โหลดและ preprocess เสร็จแล้ว

    Attributes:
        X: (จำนวนเฟรม, 8, 8, 2) — ช่อง 0 = ระยะ, ช่อง 1 = ความแรงแสง (normalize แล้ว)
        y: (จำนวนเฟรม,) — เลขคลาส 0..n-1
        groups: (จำนวนเฟรม,) — รหัสคน เช่น "User2" ใช้เป็น group ตอนแบ่งข้อมูล
        class_names: ชื่อคลาสเรียงตามเลข index
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def parse_user_from_path(path: Path) -> str:
    """ดึงรหัสคนออกจากชื่อโฟลเดอร์ session

    Args:
        path: path ของไฟล์ .npz หรือโฟลเดอร์ session

    Returns:
        รหัสคน เช่น "User2" ถ้าหาไม่เจอคืน "Unknown"
    """
    match = _USER_PATTERN.search(str(path))
    return f"User{match.group(1)}" if match else "Unknown"


def _preprocess_session(
    zone_data: np.ndarray,
    glob_data: np.ndarray,
    max_distance: float,
    min_distance: float,
    background_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """กรองและปรับค่าข้อมูล 1 session ตามขั้นตอนของ ST

    ทำตาม data_loader.py:100-157 ทุกขั้นตอน แต่เขียนแบบ vectorize
    (ของ ST ใช้ for-loop ซ้อน ช้ากว่ามากแต่ผลลัพธ์เท่ากัน)

    ขั้นตอน:
      1. ทิ้งเฟรมที่มีค่า NaN
      2. หาโซนที่ใช้ได้ (target_status อยู่ใน 5 หรือ 9)
      3. หาจุดที่ใกล้ที่สุดในเฟรม = ตำแหน่งมือ
      4. ทิ้งทั้งเฟรมถ้ามืออยู่นอกช่วง min..max (มือไม่อยู่ในระยะที่สนใจ)
      5. ตัดโซนที่ลึกกว่ามือเกิน background_distance ทิ้ง (= ฉากหลัง ไม่ใช่มือ)
      6. โซนที่ใช้ไม่ได้ ใส่ค่าเริ่มต้นแทน

    Args:
        zone_data: (4, 64, จำนวนเฟรม) จากไฟล์ .npz
        glob_data: (1, จำนวนเฟรม) ค่า label ดิบ
        max_distance: ระยะไกลสุดที่ยอมรับ (มม.)
        min_distance: ระยะใกล้สุดที่ยอมรับ (มม.)
        background_distance: ความลึกจากมือที่ยังนับว่าเป็นมือ (มม.)

    Returns:
        (X, y) โดย X = (จำนวนเฟรมที่เหลือ, 8, 8, 2), y = (จำนวนเฟรมที่เหลือ,) label ดิบ
    """
    # --- 1. ทิ้งเฟรมที่มี NaN ---
    nan_mask = np.isnan(zone_data).any(axis=(0, 1)) | np.isnan(glob_data).any(axis=0)
    zone = zone_data[:, :, ~nan_mask].copy()
    glob = glob_data[:, ~nan_mask].copy()

    if zone.shape[2] == 0:
        return np.empty((0, 8, 8, 2)), np.empty((0,))

    status = zone[CH_TARGET_STATUS]  # (64, เฟรม)
    dist = zone[CH_DISTANCE]         # (64, เฟรม)
    signal = zone[CH_SIGNAL]         # (64, เฟรม)

    # --- 2. โซนไหนใช้ได้บ้าง ---
    valid = np.isin(status, VALID_STATUS)  # (64, เฟรม) True/False

    # --- 3. จุดใกล้ที่สุดของแต่ละเฟรม = ตำแหน่งมือ ---
    # โซนที่ใช้ไม่ได้ ตั้งเป็น 4000 ไปก่อน จะได้ไม่ถูกเลือกเป็นค่าต่ำสุด
    dist_for_min = np.where(valid, dist, DEFAULT_DISTANCE)
    hand_dist = dist_for_min.min(axis=0)  # (เฟรม,)

    # --- 4. เฟรมที่มืออยู่นอกช่วง = ทิ้งทั้งเฟรม ---
    in_range = (hand_dist >= min_distance) & (hand_dist <= max_distance)

    # --- 5. ตัดฉากหลัง: โซนที่ลึกกว่ามือเกินค่ากำหนด ไม่ใช่มือ ---
    valid &= dist <= (hand_dist[np.newaxis, :] + background_distance)

    # --- 6. โซนที่ใช้ไม่ได้ ใส่ค่าเริ่มต้น ---
    dist_out = np.where(valid, dist, DEFAULT_DISTANCE)
    sig_out = np.where(valid, signal, DEFAULT_SIGNAL)

    # เก็บเฉพาะเฟรมที่ผ่านการกรอง
    dist_out = dist_out[:, in_range]
    sig_out = sig_out[:, in_range]
    y_raw = glob[0, in_range]

    if dist_out.shape[1] == 0:
        return np.empty((0, 8, 8, 2)), np.empty((0,))

    # --- normalize (data_loader.py:156-157) ---
    dist_norm = (dist_out - DIST_MEAN) / DIST_STD
    sig_norm = (sig_out - SIG_MEAN) / SIG_STD

    # จัดรูปเป็น (เฟรม, 8, 8, 2) ให้พร้อมป้อนเข้า CNN
    n_frames = dist_norm.shape[1]
    X = np.stack(
        [
            dist_norm.T.reshape(n_frames, 8, 8),
            sig_norm.T.reshape(n_frames, 8, 8),
        ],
        axis=-1,
    )
    return X, y_raw


def load_dataset(
    data_dir: str | Path,
    class_names: list[str] | None = None,
    max_distance: float = MAX_DISTANCE_MM,
    min_distance: float = MIN_DISTANCE_MM,
    background_distance: float = BACKGROUND_DISTANCE_MM,
    verbose: bool = True,
) -> HandPostureData:
    """โหลดชุดข้อมูลทั้งหมด พร้อมเก็บรหัสคนไว้ทำ subject-independent split

    Args:
        data_dir: โฟลเดอร์ ST_VL53L8CX_handposture_dataset
        class_names: รายชื่อคลาสที่ต้องการ (None = เอาทุกคลาสที่เจอ)
        max_distance: ระยะไกลสุด (มม.)
        min_distance: ระยะใกล้สุด (มม.)
        background_distance: ระยะตัดฉากหลัง (มม.)
        verbose: พิมพ์รายงานระหว่างโหลดหรือไม่

    Returns:
        HandPostureData

    Raises:
        FileNotFoundError: หาโฟลเดอร์ไม่เจอ หรือไม่มีไฟล์ .npz เลย
        ValueError: ระบุคลาสที่ไม่มีในชุดข้อมูล
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ชุดข้อมูล: {root}")

    found_classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not found_classes:
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาสใน: {root}")

    if class_names is None:
        class_names = found_classes
    else:
        missing = set(class_names) - set(found_classes)
        if missing:
            raise ValueError(f"ไม่พบคลาสเหล่านี้ในชุดข้อมูล: {sorted(missing)}")

    # เรียงคลาสตามเลข label ของ ST เพื่อให้ index ตรงกับของเขา
    class_names = sorted(class_names, key=lambda c: HAND_POSTURE_DICT[c])
    label_to_index = {HAND_POSTURE_DICT[c]: i for i, c in enumerate(class_names)}

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    n_raw_total = 0

    for cls in class_names:
        npz_files = sorted((root / cls).glob("*/npz/*.npz"))
        if not npz_files and verbose:
            print(f"  คำเตือน: คลาส {cls} ไม่มีไฟล์ .npz")

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                zone_data = data["zone_data"]
                glob_data = data["glob_data"]

            n_raw_total += zone_data.shape[2]
            X, y_raw = _preprocess_session(
                zone_data, glob_data, max_distance, min_distance, background_distance
            )
            if len(y_raw) == 0:
                continue

            # แปลง label ดิบ (0/20/21/...) เป็น index 0..n-1
            keep = np.isin(y_raw, list(label_to_index))
            if not keep.any():
                continue
            y_idx = np.array([label_to_index[int(v)] for v in y_raw[keep]])

            X_list.append(X[keep])
            y_list.append(y_idx)
            group_list.append(np.full(keep.sum(), parse_user_from_path(npz_path)))

    if not X_list:
        raise FileNotFoundError("โหลดข้อมูลไม่ได้เลยสักเฟรม — ตรวจ path และค่ากรอง")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)
    groups_all = np.concatenate(group_list, axis=0)

    if verbose:
        kept = len(y_all)
        print(f"เฟรมดิบทั้งหมด : {n_raw_total:,}")
        print(f"เฟรมหลังกรอง   : {kept:,}  ({kept / n_raw_total * 100:.1f}% ของเดิม)")
        print(f"ถูกกรองทิ้ง     : {n_raw_total - kept:,}")
        print(f"รูปร่างข้อมูล X : {X_all.shape}  (เฟรม, 8, 8, [ระยะ, ความแรงแสง])")

    return HandPostureData(X_all, y_all, groups_all, class_names)


def summarise(data: HandPostureData) -> None:
    """พิมพ์ตารางสรุปจำนวนเฟรมแยกตามคลาสและตามคน

    ใช้ตรวจว่าข้อมูลสมดุลพอจะทำ subject-independent split หรือไม่
    """
    # .tolist() แปลง np.str_ เป็น str ธรรมดา เพื่อให้พิมพ์ออกมาอ่านง่าย
    users = sorted(set(data.groups.tolist()))

    print("\n--- จำนวนเฟรม: คลาส x คน ---")
    header = f"{'คลาส':<14}" + "".join(f"{u:>9}" for u in users) + f"{'รวม':>9}"
    print(header)
    print("-" * len(header))

    for i, cls in enumerate(data.class_names):
        mask_cls = data.y == i
        row = f"{cls:<14}"
        for u in users:
            row += f"{int((mask_cls & (data.groups == u)).sum()):>9,}"
        row += f"{int(mask_cls.sum()):>9,}"
        print(row)

    total_row = f"{'รวม':<14}"
    for u in users:
        total_row += f"{int((data.groups == u).sum()):>9,}"
    total_row += f"{len(data):>9,}"
    print("-" * len(header))
    print(total_row)

    # เตือนคลาสที่มีคนทำน้อยเกินไป — จะทำ subject-independent ไม่ได้
    print()
    for i, cls in enumerate(data.class_names):
        n_users = sum(1 for u in users if ((data.y == i) & (data.groups == u)).any())
        if n_users < 2:
            print(
                f"  เตือน: คลาส '{cls}' มีข้อมูลจากคนเดียว "
                f"-> ถ้าคนนั้นไปอยู่ฝั่ง test โมเดลจะไม่เคยเห็นคลาสนี้ตอน train"
            )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="โหลดและสรุปชุดข้อมูล ST_VL53L8CX_handposture_dataset"
    )
    parser.add_argument(
        "--data-dir", required=True, help="path ของโฟลเดอร์ ST_VL53L8CX_handposture_dataset"
    )
    parser.add_argument(
        "--drop-none",
        action="store_true",
        help="ตัดคลาส None ออก (คลาสนี้มีข้อมูลจาก User1 คนเดียว)",
    )
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE_MM)
    parser.add_argument("--min-distance", type=float, default=MIN_DISTANCE_MM)
    parser.add_argument("--background-distance", type=float, default=BACKGROUND_DISTANCE_MM)
    args = parser.parse_args()

    classes = None
    if args.drop_none:
        classes = [c for c in HAND_POSTURE_DICT if c != "None"]
        classes = [c for c in classes if (Path(args.data_dir) / c).is_dir()]

    data = load_dataset(
        args.data_dir,
        class_names=classes,
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        background_distance=args.background_distance,
    )
    summarise(data)

    print("\n--- สัดส่วนคลาส (ดูว่าข้อมูลเอียงไหม) ---")
    counts = Counter(data.y.tolist())
    for i, cls in enumerate(data.class_names):
        pct = counts[i] / len(data) * 100
        print(f"  {cls:<14} {counts[i]:>6,}  {pct:5.1f}%  {'#' * int(pct / 2)}")


if __name__ == "__main__":
    _main()

# ─────────────────────────────────────────────────────────────────────────
# ข้อจำกัดที่ทราบ (failure modes)
#
# 1. ค่า normalization (295/196, 281/452) เป็นค่าคงที่ที่ ST คำนวณจากข้อมูลของเขา
#    ถ้าเราเก็บข้อมูลเองในอนาคต ต้องตรวจว่าค่านี้ยังเหมาะสมหรือไม่
#
# 2. โค้ดนี้เขียนแบบ vectorize ต่างจาก ST ที่ใช้ for-loop
#    ตรรกะเหมือนกันแต่ควรตรวจสอบผลให้ตรงกันก่อนนำไปอ้างอิงในวิทยานิพนธ์
#
# 3. เฟรมที่ไม่เหลือโซนที่ใช้ได้เลย จะได้ระยะ 4000 ทุกโซน
#    ยังไม่ได้ตัดออก เพราะ ST ก็ไม่ได้ตัด — ควรตรวจว่ามีกี่เฟรม
#
# 4. คลาส None มีข้อมูลจาก User1 คนเดียว ทำ subject-independent ไม่ได้
#    ใช้ --drop-none เพื่อตัดออก
# ─────────────────────────────────────────────────────────────────────────
