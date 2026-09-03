"""
verify_against_st.py
====================
พิสูจน์ว่า st_dataset.py ของเรา ให้ผลลัพธ์ตรงกับ data_loader.py ของ ST ทุกตัวเลข

ทำไมต้องพิสูจน์:
  st_dataset.py เขียนแบบ vectorize (คำนวณทั้งก้อนพร้อมกัน)
  แต่ ST เขียนแบบ for-loop ซ้อน (วนทีละโซน ทีละเฟรม)
  ตรรกะ "ควรจะ" เหมือนกัน แต่คำว่า "ควรจะ" ใช้อ้างในวิทยานิพนธ์ไม่ได้
  ต้องพิสูจน์ด้วยตัวเลขจริง

วิธีพิสูจน์:
  1. คัดลอก for-loop ของ ST มาทีละบรรทัด (ฟังก์ชัน st_reference_preprocess)
     ไม่แก้ไขตรรกะใดๆ ทั้งสิ้น -- แก้แค่ให้รันแยกได้โดยไม่ต้องมี TensorFlow
  2. รันทั้งสองเวอร์ชันบนไฟล์ .npz เดียวกันทุกไฟล์
  3. เทียบผลลัพธ์ทีละตัวเลข ต้องตรงกัน 100%

ที่มาของโค้ดอ้างอิง:
  stm32ai-modelzoo-services/hand_posture/tf/src/preprocessing/data_loader.py
  บรรทัด 100-157

วิธีรัน:
    python verify_against_st.py --data-dir ./ST_VL53L8CX_handposture_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from st_dataset import (
    BACKGROUND_DISTANCE_MM,
    CH_DISTANCE,
    CH_SIGNAL,
    CH_TARGET_STATUS,
    CH_VALID,
    DEFAULT_DISTANCE,
    DEFAULT_SIGNAL,
    DIST_MEAN,
    DIST_STD,
    MAX_DISTANCE_MM,
    MIN_DISTANCE_MM,
    SIG_MEAN,
    SIG_STD,
    VALID_STATUS,
    _preprocess_session,
)


def st_reference_preprocess(
    zone_data: np.ndarray,
    glob_data: np.ndarray,
    max_distance: float,
    min_distance: float,
    background_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """โค้ดอ้างอิง — คัดลอกตรรกะจาก ST มาทีละบรรทัด ห้ามปรับปรุง

    ตั้งใจเขียนให้ช้าและอ่านยากเหมือนต้นฉบับ เพราะจุดประสงค์คือ
    "ทำเหมือนเดิมทุกประการ" ไม่ใช่ "ทำให้ดีขึ้น"

    Args:
        zone_data: (4, 64, จำนวนเฟรม) จากไฟล์ .npz
        glob_data: (1, จำนวนเฟรม) label ดิบ
        max_distance: ระยะไกลสุด (มม.)
        min_distance: ระยะใกล้สุด (มม.)
        background_distance: ระยะตัดฉากหลัง (มม.)

    Returns:
        (X, y) โดย X = (เฟรมที่เหลือ, 8, 8, 2) normalize แล้ว, y = label ดิบ
    """
    X0 = zone_data.copy()
    y0 = glob_data.copy()

    # --- ST บรรทัด 102-105: ทิ้งเฟรมที่มี NaN ---
    zone_nan_mask = np.isnan(X0).any(axis=(0, 1))
    glob_nan_mask = np.isnan(y0).any(axis=0)
    nan_mask = np.logical_or(zone_nan_mask, glob_nan_mask)
    X1 = X0[:, :, ~nan_mask]
    y1 = y0[:, ~nan_mask]

    if X1.shape[2] == 0:
        return np.empty((0, 8, 8, 2)), np.empty((0,))

    # --- ST บรรทัด 113-138: กรองระยะและตัดฉากหลัง (for-loop ตามต้นฉบับ) ---
    (max_fields, max_zones, max_frames) = X1.shape
    index_GestureGT = 0
    index_distance = CH_DISTANCE
    index_status = CH_TARGET_STATUS
    index_valid = CH_VALID
    valid_status = list(VALID_STATUS)

    for i in range(max_frames):
        min_val = 4000
        for j in range(max_zones):
            X1[index_valid, j, i] = 0
            # โซนใช้ได้ ถ้า target_status อยู่ในรายการ
            if X1[index_status, j, i] in valid_status:
                X1[index_valid, j, i] = 1
                if X1[index_distance, j, i] < min_val:
                    min_val = X1[index_distance, j, i]
        # มืออยู่นอกช่วง -> ทำเครื่องหมายทิ้งทั้งเฟรม
        if min_val > max_distance or min_val < min_distance:
            y1[index_GestureGT, i] = np.nan
        # ตัดสิ่งที่อยู่ลึกกว่าจุดใกล้สุดเกินกำหนด
        for j in range(max_zones):
            if X1[index_valid, j, i] == 1:
                if X1[index_distance, j, i] > (min_val + background_distance):
                    X1[index_valid, j, i] = 0

    # --- ST บรรทัด 140-147: ลบเฟรมที่ทำเครื่องหมายไว้ ---
    zone_nan_mask = np.isnan(X1).any(axis=(0, 1))
    glob_nan_mask = np.isnan(y1).any(axis=0)
    nan_mask = np.logical_or(zone_nan_mask, glob_nan_mask)
    X1 = X1[:, :, ~nan_mask]
    y1 = y1[:, ~nan_mask]

    if X1.shape[2] == 0:
        return np.empty((0, 8, 8, 2)), np.empty((0,))

    # --- ST บรรทัด 150-153: โซนที่ใช้ไม่ได้ ใส่ค่าเริ่มต้น ---
    np.place(X1[CH_DISTANCE], X1[index_valid] == 0, DEFAULT_DISTANCE)
    np.place(X1[CH_SIGNAL], X1[index_valid] == 0, DEFAULT_SIGNAL)

    # --- ST บรรทัด 155-157: normalize ---
    dist_norm = (X1[CH_DISTANCE] - DIST_MEAN) / DIST_STD
    sig_norm = (X1[CH_SIGNAL] - SIG_MEAN) / SIG_STD

    # --- ST บรรทัด 170-172: จัดรูปเป็น (เฟรม, 8, 8, ช่อง) ---
    n_frames = dist_norm.shape[1]
    X_out = np.stack(
        [dist_norm.T.reshape(n_frames, 8, 8), sig_norm.T.reshape(n_frames, 8, 8)],
        axis=-1,
    )
    return X_out, y1[index_GestureGT]


def verify(data_dir: str | Path, verbose: bool = True) -> bool:
    """รันทั้งสองเวอร์ชันบนทุกไฟล์ แล้วเทียบผลทีละตัวเลข

    Args:
        data_dir: โฟลเดอร์ ST_VL53L8CX_handposture_dataset
        verbose: พิมพ์รายละเอียดระหว่างตรวจ

    Returns:
        True ถ้าตรงกันทุกไฟล์

    Raises:
        FileNotFoundError: ไม่พบไฟล์ .npz
    """
    root = Path(data_dir)
    npz_files = sorted(root.glob("*/*/npz/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ .npz ใน {root}")

    n_ok = 0
    mismatches: list[tuple[str, str]] = []
    total_raw = 0
    total_ours = 0
    total_st = 0

    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as d:
            zone_data = d["zone_data"]
            glob_data = d["glob_data"]

        total_raw += zone_data.shape[2]

        # สำเนาแยกกัน เพราะโค้ด ST แก้ไข array ในที่ (in-place)
        X_ours, y_ours = _preprocess_session(
            zone_data.copy(), glob_data.copy(),
            MAX_DISTANCE_MM, MIN_DISTANCE_MM, BACKGROUND_DISTANCE_MM,
        )
        X_st, y_st = st_reference_preprocess(
            zone_data.copy(), glob_data.copy(),
            MAX_DISTANCE_MM, MIN_DISTANCE_MM, BACKGROUND_DISTANCE_MM,
        )

        total_ours += len(y_ours)
        total_st += len(y_st)

        name = npz_path.parent.parent.name

        if X_ours.shape != X_st.shape:
            mismatches.append((name, f"รูปร่างต่างกัน: เรา {X_ours.shape} vs ST {X_st.shape}"))
            continue
        if not np.array_equal(y_ours, y_st):
            mismatches.append((name, "label ไม่ตรงกัน"))
            continue
        if not np.allclose(X_ours, X_st, rtol=0, atol=0, equal_nan=True):
            diff = np.nanmax(np.abs(X_ours - X_st))
            mismatches.append((name, f"ค่าข้อมูลต่างกันสูงสุด {diff:.3e}"))
            continue

        n_ok += 1

    if verbose:
        print(f"ไฟล์ที่ตรวจ      : {len(npz_files)}")
        print(f"ตรงกันเป๊ะ       : {n_ok}")
        print(f"ไม่ตรง           : {len(mismatches)}")
        print()
        print(f"เฟรมดิบ          : {total_raw:,}")
        print(f"เฟรมเหลือ (เรา)  : {total_ours:,}")
        print(f"เฟรมเหลือ (ST)   : {total_st:,}")
        print(f"ถูกกรองทิ้ง      : {total_raw - total_st:,}")

        if mismatches:
            print("\n--- รายการที่ไม่ตรง ---")
            for name, reason in mismatches[:20]:
                print(f"  {name}: {reason}")

    return len(mismatches) == 0


def count_zone_removal(data_dir: str | Path) -> None:
    """นับว่ามีการตัด "โซน" ทิ้งไปกี่ % (คนละเรื่องกับการตัด "เฟรม")

    เขียนฟังก์ชันนี้เพราะบันทึกใน PROJECT_FACTS §5.1 เขียนว่า
    "most of the 11,448 raw frames are filtered out"
    แต่วัดจริงแล้วเฟรมถูกตัดน้อยมาก จึงต้องตรวจว่าคำว่า filtered
    หมายถึงการตัดโซน (background removal) หรือไม่
    """
    root = Path(data_dir)
    total_zones = 0
    removed_status = 0
    removed_background = 0

    for npz_path in sorted(root.glob("*/*/npz/*.npz")):
        with np.load(npz_path, allow_pickle=True) as d:
            z = d["zone_data"]

        keep = ~np.isnan(z).any(axis=(0, 1))
        z = z[:, :, keep]
        if z.shape[2] == 0:
            continue

        status, dist = z[CH_TARGET_STATUS], z[CH_DISTANCE]
        valid_by_status = np.isin(status, VALID_STATUS)

        hand = np.where(valid_by_status, dist, DEFAULT_DISTANCE).min(axis=0)
        valid_after_bg = valid_by_status & (dist <= hand[np.newaxis, :] + BACKGROUND_DISTANCE_MM)

        total_zones += status.size
        removed_status += int((~valid_by_status).sum())
        removed_background += int((valid_by_status & ~valid_after_bg).sum())

    kept = total_zones - removed_status - removed_background
    print(f"โซนทั้งหมด            : {total_zones:,}")
    print(f"  ตัดเพราะ status ไม่ดี : {removed_status:,}  ({removed_status / total_zones * 100:5.1f}%)")
    print(f"  ตัดเพราะเป็นฉากหลัง   : {removed_background:,}  ({removed_background / total_zones * 100:5.1f}%)")
    print(f"  เหลือใช้จริง          : {kept:,}  ({kept / total_zones * 100:5.1f}%)")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="พิสูจน์ว่า st_dataset.py ให้ผลตรงกับ data_loader.py ของ ST"
    )
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    print("=" * 62)
    print("ส่วนที่ 1 — เทียบผลลัพธ์ของเรากับโค้ดอ้างอิงของ ST")
    print("=" * 62)
    ok = verify(args.data_dir)

    print()
    print("=" * 62)
    print("ส่วนที่ 2 — แยกให้ชัด: ตัด 'เฟรม' กับตัด 'โซน' คนละเรื่องกัน")
    print("=" * 62)
    count_zone_removal(args.data_dir)

    print()
    if ok:
        print("ผลสรุป: ตรงกันทุกไฟล์ ทุกตัวเลข — loader ของเราใช้อ้างอิงได้")
    else:
        print("ผลสรุป: มีจุดไม่ตรง ต้องแก้ก่อนใช้งานต่อ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()

# ─────────────────────────────────────────────────────────────────────────
# ข้อจำกัดที่ทราบ
#
# 1. โค้ดอ้างอิงคัดลอกเฉพาะส่วน preprocessing (บรรทัด 100-172)
#    ไม่รวมส่วนแปลง label และส่วนแบ่งข้อมูล เพราะสองส่วนนั้นเราตั้งใจทำต่างจาก ST
#
# 2. เทียบด้วย atol=0 rtol=0 คือต้องเท่ากันทุกบิต
#    ถ้าอนาคตเปลี่ยนลำดับการคำนวณ อาจต้องผ่อนเป็น atol=1e-12
# ─────────────────────────────────────────────────────────────────────────
