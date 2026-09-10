"""
analyze_bus.py
==============
วิเคราะห์สัญญาณบัส I2C และขา INT จากไฟล์ที่ sigrok-cli บันทึกไว้

ตอบสามคำถาม
-----------
H3  ในหนึ่งคาบเฟรม บัส I2C ว่างกี่เปอร์เซ็นต์
    เดิมมีแต่หลักฐานทางอ้อม คือความสัมพันธ์ระหว่างการหน่วงกับคาบ
    (PROJECT_FACTS ระบุ R กำลังสอง 0.9997 จาก 6 จุด)
    การวัดนี้เป็นหลักฐานตรง เห็นกับตาว่าบัสว่างกี่มิลลิวินาที

H4  สัญญาณนาฬิกา SCL วิ่งจริงที่กี่กิโลเฮิรตซ์
    โปรเจคตั้งไว้ 400 kHz แต่ตัวหารความถี่ของ STM32 เป็นจำนวนเต็ม
    จึงมักได้ค่าที่ไม่ตรงกับที่ตั้ง ถ้าค่าจริงต่างจาก 400
    ต้องแก้ทุกที่ในวิทยานิพนธ์ที่เขียนว่า 400 kHz

H5  ขา INT ยิงห่างจากการเริ่มคุย I2C เท่าไหร่
    ใช้ตอบว่าทำไมโหมด interrupt ไม่ช่วยเพิ่มอัตราเฟรมที่ 4x4
    (PROJECT_FACTS TRAP หมายเลข 15)

ช่องสัญญาณที่คาดหวัง
--------------------
    D0 = SCL   สัญญาณนาฬิกาของบัส
    D1 = SDA   ข้อมูล
    D2 = INT   สัญญาณจากเซ็นเซอร์ บอกว่าเฟรมพร้อม

วิธีรัน
-------
    python analyze_bus.py bus4x4_run1.bin bus4x4_run2.bin bus4x4_run3.bin
    python analyze_bus.py bus8x8_run1.bin --samplerate 12000000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ต้องติดตั้ง numpy ก่อน  พิมพ์:  pip install numpy")
    sys.exit(1)


# ตัดหัวและท้ายทิ้งอย่างละเท่านี้ เพื่อเลี่ยงส่วนหัวไฟล์ของ sigrok-cli
# ไฟล์ที่ได้มีขนาดเกินจำนวนตัวอย่างที่ขอ 22 ไบต์เสมอ ไม่ว่าไฟล์ใหญ่แค่ไหน
# จึงตัดทิ้งเผื่อไว้มากกว่านั้นหลายสิบเท่า โดยไม่ต้องรู้ว่าอยู่หัวหรือท้าย
GUARD_SAMPLES: int = 2000

# ค่าเริ่มต้นของช่องสัญญาณ
CH_SCL: int = 0
CH_SDA: int = 1
CH_INT: int = 2


@dataclass
class BusResult:
    """ผลการวิเคราะห์ไฟล์หนึ่งไฟล์"""

    path: str
    duration_s: float          # ระยะเวลาที่วิเคราะห์ หน่วยวินาที

    # ---- H4 ความถี่ SCL ----
    scl_khz_median: float      # ค่ากลาง ใช้เป็นค่าหลัก
    scl_khz_mean: float
    scl_khz_min: float
    scl_khz_max: float
    n_scl_clocks: int          # จำนวนขอบขาขึ้นของ SCL ทั้งหมด

    # วัดแบบข้ามหลายพัลส์ ละเอียดกว่ามาก ใช้ค่านี้ตอบ H4
    scl_khz_long: float        # ความถี่จากการวัดข้ามหลายพัลส์
    scl_long_runs: int         # จำนวนช่วงต่อเนื่องที่ใช้
    scl_long_pulses: int       # จำนวนพัลส์ที่ใช้ทั้งหมด
    scl_long_longest: int      # ช่วงที่ยาวที่สุด
    scl_long_res_ppm: float    # ความละเอียดโดยประมาณ

    # ---- H3 บัสว่างกี่เปอร์เซ็นต์ ----
    n_bursts: int              # จำนวนชุดการคุย
    burst_ms_mean: float       # ความยาวเฉลี่ยของหนึ่งชุด
    busy_percent: float        # สัดส่วนเวลาที่บัสทำงาน
    idle_percent: float        # สัดส่วนเวลาที่บัสว่าง
    frame_period_ms: float     # คาบเฟรม วัดจากขอบขาลงของ INT (ปรับเส้นตรง)
    frame_period_se_ppm: float # ความไม่แน่นอนของคาบเฟรม หน่วย ppm
    int_jitter_us: float       # ความสั่นของขอบ INT จากส่วนตกค้าง

    # ---- H5 ระยะห่าง INT กับ I2C ----
    int_to_scl_us_median: float
    int_to_scl_us_mean: float
    int_to_scl_us_min: float
    int_to_scl_us_max: float
    n_int_events: int

    # วัดถึงจังหวะ START ซึ่งเป็นนิยามที่ถูกต้องกว่า ใช้ค่านี้ตอบ H5
    int_to_start_us_median: float
    int_to_start_us_mean: float
    int_to_start_us_min: float
    int_to_start_us_max: float

    # ---- ข้อมูลเสริม ----
    clocks_per_burst: float    # จำนวนพัลส์ SCL เฉลี่ยต่อชุด
    est_bytes_per_burst: float # ประมาณจำนวนไบต์ หารด้วย 9 เพราะมีบิต ACK
    int_pulse_us_median: float # ความกว้างพัลส์ INT


def scl_freq_long_run(
    scl_rise: np.ndarray,
    samplerate: float,
    tol: float = 0.25,
    min_run: int = 50,
) -> tuple[float, int, int, int, float]:
    """วัดความถี่ SCL แบบข้ามหลายพัลส์ เพื่อให้ละเอียดพอใช้งาน

    ทำไมต้องมีฟังก์ชันนี้
    ---------------------
    การวัดคาบทีละพัลส์มีปัญหาความละเอียด
    ที่อัตราสุ่ม 12 MHz คาบ SCL 400 kHz เท่ากับ 30 ตัวอย่างพอดี
    ตัวนับเป็นจำนวนเต็ม จึงวัดได้แค่ 29 30 หรือ 31 ตัวอย่าง
    ซึ่งแปลเป็น 413.79  400.00  387.10 kHz
    ค่าจริงที่ใดก็ได้ระหว่าง 393 ถึง 407 kHz จะถูกปัดมาเป็น 400.00 เหมือนกันหมด

    ทางแก้คือวัดระยะรวมของพัลส์ติดกันหลายพันตัว แล้วหารด้วยจำนวนพัลส์
    ความคลาดจากการปัดยังเท่าเดิมคือประมาณ 1 ตัวอย่าง
    แต่หารด้วยระยะที่ยาวขึ้นเป็นพันเท่า ความละเอียดจึงดีขึ้นเป็นพันเท่า

    เป็นหลักการเดียวกับที่ใช้วัดคลื่นสอบเทียบ 1 kHz
    ซึ่งได้ความละเอียดถึง 0.1 ppm

    ทำไมต้องคัดพัลส์
    ----------------
    ในหนึ่งชุดการคุย ไม่ใช่ทุกช่องว่างจะเท่ากัน
    ระหว่างไบต์ หรือช่วงที่อุปกรณ์ยืดสัญญาณนาฬิกา ช่องว่างจะยาวกว่าปกติ
    ถ้ารวมช่วงเหล่านั้นเข้าไปด้วย ค่าเฉลี่ยจะยาวเกินจริง ความถี่จึงต่ำเกินจริง

    จึงเก็บเฉพาะช่วงที่ช่องว่างใกล้เคียงค่ากลาง แล้วหาช่วงที่ต่อเนื่องกันยาว ๆ

    Args:
        scl_rise: ตำแหน่งขอบขาขึ้นของ SCL
        samplerate: อัตราสุ่ม หน่วย Hz
        tol: ช่องว่างที่ต่างจากค่ากลางไม่เกินสัดส่วนนี้ ถือว่าเป็นพัลส์ปกติ
        min_run: ความยาวขั้นต่ำของช่วงต่อเนื่องที่จะนำมาใช้

    Returns:
        (ความถี่ kHz, จำนวนช่วง, จำนวนพัลส์ที่ใช้, ช่วงที่ยาวที่สุด, ความละเอียด ppm)
    """
    gaps = np.diff(scl_rise)
    if gaps.size < min_run:
        return (float("nan"), 0, 0, 0, float("nan"))

    g0 = float(np.median(gaps))
    ok = np.abs(gaps - g0) <= tol * g0

    # หาช่วงที่ ok ต่อเนื่องกัน
    # เติม False หัวท้าย เพื่อให้จับขอบเริ่มและขอบจบได้ครบ
    padded = np.concatenate(([False], ok, [False]))
    change = np.diff(padded.astype(np.int8))
    run_start = np.flatnonzero(change == 1)
    run_end = np.flatnonzero(change == -1)

    total_span = 0.0
    total_n = 0
    n_runs = 0
    longest = 0

    for s, e in zip(run_start, run_end):
        length = e - s
        if length < min_run:
            continue
        # ผลรวมของช่องว่างในช่วงนี้ เท่ากับระยะจากขอบแรกถึงขอบสุดท้ายพอดี
        total_span += float(np.sum(gaps[s:e]))
        total_n += int(length)
        n_runs += 1
        longest = max(longest, int(length))

    if total_n == 0:
        return (float("nan"), 0, 0, 0, float("nan"))

    mean_period = total_span / total_n
    freq_khz = samplerate / mean_period / 1000.0

    # ความละเอียดโดยประมาณ
    # ความคลาดประมาณ 1 ตัวอย่างต่อหนึ่งช่วง รวมกันแบบรากที่สอง
    res_ppm = (np.sqrt(n_runs) / total_span) * 1e6 if total_span else float("nan")

    return (freq_khz, n_runs, total_n, longest, float(res_ppm))


def fit_period(edges: np.ndarray) -> tuple[float, float, float]:
    """หาคาบเฉลี่ยด้วยการปรับเส้นตรงกับตำแหน่งขอบทุกอัน

    ใช้วิธีเดียวกับใน analyze_calib.py เพื่อให้ตัวเลขทุกที่ในโปรเจค
    คำนวณด้วยวิธีเดียวกัน ไม่ปนกันสองวิธี

    วิธีเดิมที่ใช้เพียงขอบแรกกับขอบสุดท้าย จะรับความสั่นของสองจุดนั้นเต็มที่
    การปรับเส้นตรงใช้ขอบทุกอัน ความสั่นแบบสุ่มจึงหักล้างกัน
    วัดจริงพบว่าลดความไม่แน่นอนได้ราวสิบเท่าสำหรับสัญญาณ INT

    Args:
        edges: ตำแหน่งขอบสัญญาณ หน่วยตัวอย่าง

    Returns:
        (คาบเฉลี่ย, ความไม่แน่นอนของคาบ, รากที่สองของกำลังสองเฉลี่ยของส่วนตกค้าง)
    """
    n = edges.size
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))

    idx = np.arange(n, dtype=np.float64)
    t = edges.astype(np.float64)
    ic = idx - idx.mean()
    sxx = float(np.sum(ic * ic))
    b = float(np.sum(ic * t) / sxx)
    a = float(t.mean())
    resid = t - (a + b * ic)
    dof = n - 2
    resid_var = float(np.sum(resid * resid)) / dof
    return (b, float(np.sqrt(resid_var / sxx)), float(np.sqrt(resid_var)))


def _edges(bits: np.ndarray, rising: bool) -> np.ndarray:
    """หาตำแหน่งขอบสัญญาณ

    Args:
        bits: อาร์เรย์ 0 กับ 1
        rising: True หาขอบขาขึ้น False หาขอบขาลง

    Returns:
        ตำแหน่งตัวอย่างที่เกิดขอบ
    """
    diff = np.diff(bits.astype(np.int8))
    return np.flatnonzero(diff == (1 if rising else -1)) + 1


def find_starts(scl: np.ndarray, sda: np.ndarray) -> np.ndarray:
    """หาตำแหน่งจังหวะเริ่มการคุย (START condition) บนบัส I2C

    ทำไมต้องมี
    ----------
    เดิมวัดระยะจาก INT ถึงพัลส์ขาขึ้นแรกของ SCL
    แต่จังหวะที่ MCU ลงมือจริงคือ START ซึ่งเกิดก่อนพัลส์แรกราวหนึ่งคาบนาฬิกา
    การวัดถึงพัลส์แรกจึงให้ค่ายาวเกินจริงประมาณ 2.5 ไมโครวินาที
    เทียบกับค่าที่วัดได้ราว 6.75 ไมโครวินาที ถือว่าเกินไปราว 37 เปอร์เซ็นต์

    นิยามตามมาตรฐาน I2C
    -------------------
    START คือจังหวะที่ SDA เปลี่ยนจากสูงเป็นต่ำ ในขณะที่ SCL ยังคงสูงอยู่
    ต่างจากการส่งข้อมูลปกติ ซึ่ง SDA จะเปลี่ยนค่าเฉพาะตอนที่ SCL ต่ำเท่านั้น

    Args:
        scl: อาร์เรย์บิตของสัญญาณนาฬิกา
        sda: อาร์เรย์บิตของสัญญาณข้อมูล

    Returns:
        ตำแหน่งตัวอย่างที่เกิด START
    """
    sda_fall = _edges(sda, rising=False)
    if sda_fall.size == 0:
        return sda_fall
    # เก็บเฉพาะจุดที่ SCL ยังสูงอยู่
    return sda_fall[scl[sda_fall] == 1]


def analyse_bus(
    path: Path,
    samplerate: float,
    ch_scl: int,
    ch_sda: int,
    ch_int: int,
    gap_us: float,
) -> BusResult:
    """วิเคราะห์ไฟล์หนึ่งไฟล์

    Args:
        path: ไฟล์ binary จาก sigrok-cli
        samplerate: อัตราสุ่มที่ตั้งไว้ หน่วย Hz
        ch_scl: หมายเลขบิตของช่อง SCL
        ch_sda: หมายเลขบิตของช่อง SDA
        ch_int: หมายเลขบิตของช่อง INT
        gap_us: ช่องว่างระหว่างขอบ SCL ที่เกินค่านี้ ถือว่าจบชุดการคุย

    Returns:
        BusResult

    Raises:
        FileNotFoundError: หาไฟล์ไม่เจอ
        ValueError: ข้อมูลไม่พอ หรือไม่พบสัญญาณที่คาดหวัง
    """
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 10 * GUARD_SAMPLES:
        raise ValueError(f"ข้อมูลน้อยเกินไป มีเพียง {raw.size} ไบต์")

    core = raw[GUARD_SAMPLES:-GUARD_SAMPLES]
    n = core.size
    duration_s = n / samplerate

    scl = (core >> ch_scl) & 1
    int_ = (core >> ch_int) & 1

    # ==== H4 ความถี่ของ SCL ====
    #
    # วัดจากขอบขาขึ้นถึงขอบขาขึ้นถัดไป
    # แต่ต้องคัดเฉพาะช่วงที่อยู่ในชุดการคุยเดียวกัน
    # เพราะช่องว่างระหว่างชุดยาวกว่ามาก จะทำให้ค่าเฉลี่ยเพี้ยน
    #
    # ใช้ค่ากลางแทนค่าเฉลี่ย เพราะทนต่อค่าผิดปกติได้ดีกว่า
    # ช่องว่างระหว่างไบต์และระหว่างชุดจะกลายเป็นค่าสุดโต่ง
    # ซึ่งค่ากลางไม่สนใจ แต่ค่าเฉลี่ยจะถูกดึง
    scl_rise = _edges(scl, rising=True)
    if scl_rise.size < 10:
        raise ValueError(
            f"พบขอบขาขึ้นของ SCL เพียง {scl_rise.size} จุด "
            "ตรวจว่าต่อสายช่อง SCL ถูกหรือไม่"
        )

    scl_gaps = np.diff(scl_rise)
    gap_samples = gap_us * samplerate / 1e6

    # เก็บเฉพาะช่องว่างที่สั้น ซึ่งคือพัลส์นาฬิกาที่ติดกันจริง
    in_burst = scl_gaps[scl_gaps < gap_samples]
    if in_burst.size < 10:
        raise ValueError("ไม่พบพัลส์นาฬิกาที่ติดกันพอจะวัดความถี่ได้")

    scl_khz_median = samplerate / float(np.median(in_burst)) / 1000.0
    scl_khz_mean = samplerate / float(np.mean(in_burst)) / 1000.0
    scl_khz_min = samplerate / float(np.max(in_burst)) / 1000.0
    scl_khz_max = samplerate / float(np.min(in_burst)) / 1000.0

    # วัดแบบข้ามหลายพัลส์ ใช้ค่านี้เป็นคำตอบของ H4
    (scl_khz_long, scl_long_runs, scl_long_pulses,
     scl_long_longest, scl_long_res_ppm) = scl_freq_long_run(scl_rise, samplerate)

    # ==== H3 บัสว่างกี่เปอร์เซ็นต์ ====
    #
    # แบ่งขอบ SCL ออกเป็นชุด โดยใช้ช่องว่างที่ยาวเป็นตัวแบ่ง
    # ความยาวของชุด = ขอบสุดท้ายลบขอบแรกของชุดนั้น
    # เวลาที่บัสทำงาน = ผลรวมความยาวของทุกชุด
    split_at = np.flatnonzero(scl_gaps >= gap_samples)

    starts = np.concatenate(([scl_rise[0]], scl_rise[split_at + 1]))
    ends = np.concatenate((scl_rise[split_at], [scl_rise[-1]]))

    burst_len = ends - starts
    n_bursts = int(starts.size)
    busy_samples = float(np.sum(burst_len))
    busy_percent = busy_samples / n * 100.0

    clocks_in_burst = np.concatenate(
        ([split_at[0] + 1] if split_at.size else [scl_rise.size],
         np.diff(split_at) if split_at.size > 1 else [],
         [scl_rise.size - split_at[-1] - 1] if split_at.size else [])
    ).astype(float)

    # ==== คาบเฟรม วัดจากขอบขาลงของ INT ====
    #
    # ขอบขาลงคือจังหวะที่เซ็นเซอร์ยิงบอกว่าเฟรมพร้อม
    int_fall = _edges(int_, rising=False)
    int_rise = _edges(int_, rising=True)

    if int_fall.size >= 3:
        per_fit, se_fit, jit = fit_period(int_fall)
        frame_period_ms = per_fit / samplerate * 1000.0
        frame_period_se_ppm = se_fit / per_fit * 1e6
        int_jitter_us = jit / samplerate * 1e6
    else:
        frame_period_ms = float("nan")
        frame_period_se_ppm = float("nan")
        int_jitter_us = float("nan")

    # ความกว้างพัลส์ INT จับคู่ขอบขาลงกับขอบขาขึ้นที่ตามมา
    widths = []
    for f in int_fall:
        after = int_rise[int_rise > f]
        if after.size:
            widths.append(after[0] - f)
    int_pulse_us_median = (
        float(np.median(widths)) / samplerate * 1e6 if widths else float("nan")
    )

    # ==== H5 ระยะจากขอบขาลงของ INT ถึงพัลส์ SCL แรกที่ตามมา ====
    lat = []
    for f in int_fall:
        after = scl_rise[scl_rise > f]
        if after.size:
            lat.append(after[0] - f)
    # วัดถึง START ซึ่งเป็นจังหวะที่ MCU ลงมือจริง
    sda = (core >> ch_sda) & 1
    starts = find_starts(scl, sda)
    lat_s = []
    for f in int_fall:
        after = starts[starts > f]
        if after.size:
            lat_s.append(after[0] - f)
    if lat_s:
        ls = np.array(lat_s, dtype=float) / samplerate * 1e6
        int_to_start_us_median = float(np.median(ls))
        int_to_start_us_mean = float(np.mean(ls))
        int_to_start_us_min = float(np.min(ls))
        int_to_start_us_max = float(np.max(ls))
    else:
        int_to_start_us_median = int_to_start_us_mean = float("nan")
        int_to_start_us_min = int_to_start_us_max = float("nan")

    if lat:
        lat_us = np.array(lat, dtype=float) / samplerate * 1e6
        int_to_scl_us_median = float(np.median(lat_us))
        int_to_scl_us_mean = float(np.mean(lat_us))
        int_to_scl_us_min = float(np.min(lat_us))
        int_to_scl_us_max = float(np.max(lat_us))
    else:
        int_to_scl_us_median = int_to_scl_us_mean = float("nan")
        int_to_scl_us_min = int_to_scl_us_max = float("nan")

    cpb = float(np.mean(clocks_in_burst)) if clocks_in_burst.size else float("nan")

    return BusResult(
        path=str(path),
        duration_s=duration_s,
        scl_khz_median=scl_khz_median,
        scl_khz_mean=scl_khz_mean,
        scl_khz_min=scl_khz_min,
        scl_khz_max=scl_khz_max,
        n_scl_clocks=int(scl_rise.size),
        scl_khz_long=scl_khz_long,
        scl_long_runs=scl_long_runs,
        scl_long_pulses=scl_long_pulses,
        scl_long_longest=scl_long_longest,
        scl_long_res_ppm=scl_long_res_ppm,
        n_bursts=n_bursts,
        burst_ms_mean=float(np.mean(burst_len)) / samplerate * 1000.0,
        busy_percent=busy_percent,
        idle_percent=100.0 - busy_percent,
        frame_period_ms=frame_period_ms,
        frame_period_se_ppm=frame_period_se_ppm,
        int_jitter_us=int_jitter_us,
        int_to_scl_us_median=int_to_scl_us_median,
        int_to_scl_us_mean=int_to_scl_us_mean,
        int_to_scl_us_min=int_to_scl_us_min,
        int_to_scl_us_max=int_to_scl_us_max,
        n_int_events=int(int_fall.size),
        int_to_start_us_median=int_to_start_us_median,
        int_to_start_us_mean=int_to_start_us_mean,
        int_to_start_us_min=int_to_start_us_min,
        int_to_start_us_max=int_to_start_us_max,
        clocks_per_burst=cpb,
        est_bytes_per_burst=cpb / 9.0,
        int_pulse_us_median=int_pulse_us_median,
    )


def report(res: BusResult) -> None:
    """พิมพ์ผลของไฟล์เดียว"""
    print(f"\n--- {Path(res.path).name} ---")
    print(f"  ระยะเวลาที่วิเคราะห์ : {res.duration_s:.4f} วินาที")
    print()
    print("  [H4] ความถี่สัญญาณนาฬิกา SCL")
    print(f"    วัดข้ามหลายพัลส์    : {res.scl_khz_long:8.4f} kHz  <- ใช้ค่านี้")
    print(f"      ช่วงที่ใช้         : {res.scl_long_runs:,} ช่วง  "
          f"{res.scl_long_pulses:,} พัลส์  ยาวสุด {res.scl_long_longest:,}")
    print(f"      ความละเอียด       : {res.scl_long_res_ppm:8.2f} ppm")
    print(f"    วัดทีละพัลส์ (หยาบ) : {res.scl_khz_median:8.2f} kHz")
    print(f"    ค่าเฉลี่ย           : {res.scl_khz_mean:8.2f} kHz")
    print(f"    ต่ำสุด / สูงสุด     : {res.scl_khz_min:.2f} / {res.scl_khz_max:.2f} kHz")
    print(f"    จำนวนพัลส์ทั้งหมด   : {res.n_scl_clocks:,}")
    print()
    print("  [H3] การใช้งานบัส")
    print(f"    คาบเฟรม            : {res.frame_period_ms:8.4f} ms "
          f"(+-{res.frame_period_se_ppm:.2f} ppm, ปรับเส้นตรง)")
    print(f"    ความสั่นขอบ INT     : {res.int_jitter_us:8.1f} us")
    print(f"    จำนวนชุดการคุย      : {res.n_bursts:,}")
    print(f"    ความยาวชุดเฉลี่ย    : {res.burst_ms_mean:8.4f} ms")
    print(f"    บัสทำงาน           : {res.busy_percent:8.3f} %")
    print(f"    บัสว่าง            : {res.idle_percent:8.3f} %   <- ตอบ H3")
    print(f"    พัลส์ต่อชุด         : {res.clocks_per_burst:8.1f}")
    print(f"    ประมาณไบต์ต่อชุด    : {res.est_bytes_per_burst:8.1f}  (หาร 9 เพราะมีบิต ACK)")
    print()
    print("  [H5] ระยะจากขอบขาลงของ INT ถึงจังหวะเริ่มการคุย (START)")
    print(f"    ค่ากลาง            : {res.int_to_start_us_median:8.2f} us   <- ตอบ H5")
    print(f"    ค่าเฉลี่ย           : {res.int_to_start_us_mean:8.2f} us")
    print(f"    ต่ำสุด / สูงสุด     : {res.int_to_start_us_min:.2f} / {res.int_to_start_us_max:.2f} us")
    print()
    print("  [เทียบ] ระยะถึงพัลส์ SCL แรก (ยาวกว่าราวหนึ่งคาบนาฬิกา)")
    print(f"    ค่ากลาง            : {res.int_to_scl_us_median:8.2f} us")
    print(f"    ค่าเฉลี่ย           : {res.int_to_scl_us_mean:8.2f} us")
    print(f"    ต่ำสุด / สูงสุด     : {res.int_to_scl_us_min:.2f} / {res.int_to_scl_us_max:.2f} us")
    print(f"    จำนวนครั้งที่วัดได้  : {res.n_int_events:,}")
    print()
    print(f"  ความกว้างพัลส์ INT   : {res.int_pulse_us_median:8.2f} us")


def compare(results: list[BusResult]) -> None:
    """เทียบผลหลายรอบ ตามกฎที่ว่า n น้อยกว่า 3 ห้ามสรุป"""
    print("\n" + "=" * 66)
    print("เทียบผลระหว่างรอบ")
    print("=" * 66)

    if len(results) < 3:
        print(f"  เตือน: มีเพียง {len(results)} รอบ")
        print("  ตามกฎของโปรเจค ต้องมีอย่างน้อย 3 รอบจึงจะสรุปได้")

    print(f"\n  {'ไฟล์':<20}{'SCL kHz':>10}{'ว่าง %':>10}{'INT->START us':>14}{'คาบ ms':>11}")
    print("  " + "-" * 63)
    for r in results:
        print(
            f"  {Path(r.path).name:<20}"
            f"{r.scl_khz_long:>10.4f}"
            f"{r.idle_percent:>10.3f}"
            f"{r.int_to_start_us_median:>14.2f}"
            f"{r.frame_period_ms:>11.4f}"
        )

    if len(results) >= 2:
        arrs = {
            "SCL kHz": np.array([r.scl_khz_long for r in results]),
            "ว่าง %": np.array([r.idle_percent for r in results]),
            "INT->SCL us": np.array([r.int_to_start_us_median for r in results]),
            "คาบ ms": np.array([r.frame_period_ms for r in results]),
        }
        print("  " + "-" * 63)
        line_mean = "  " + f"{'ค่าเฉลี่ย':<20}"
        line_sd = "  " + f"{'ส่วนเบี่ยงเบน':<20}"
        for key, width in zip(arrs, (10, 10, 14, 11)):
            line_mean += f"{arrs[key].mean():>{width}.4f}"
            line_sd += f"{arrs[key].std():>{width}.4f}"
        print(line_mean)
        print(line_sd)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="วิเคราะห์บัส I2C และขา INT จากไฟล์ binary ของ sigrok-cli"
    )
    parser.add_argument("files", nargs="+", help="ไฟล์ .bin ที่ต้องการวิเคราะห์")
    parser.add_argument(
        "--samplerate",
        type=float,
        default=12_000_000.0,
        help="อัตราสุ่มที่ตั้งไว้ตอนเก็บข้อมูล หน่วย Hz ค่าเริ่มต้น 12000000",
    )
    parser.add_argument("--ch-scl", type=int, default=CH_SCL, help="ช่อง SCL ค่าเริ่มต้น 0")
    parser.add_argument("--ch-sda", type=int, default=CH_SDA, help="ช่อง SDA ค่าเริ่มต้น 1")
    parser.add_argument("--ch-int", type=int, default=CH_INT, help="ช่อง INT ค่าเริ่มต้น 2")
    parser.add_argument(
        "--gap-us",
        type=float,
        default=100.0,
        help="ช่องว่างระหว่างพัลส์ SCL ที่เกินค่านี้ ถือว่าจบชุดการคุย "
             "ค่าเริ่มต้น 100 ซึ่งยาวกว่าคาบนาฬิกา 2.5 us มาก "
             "แต่สั้นกว่าคาบเฟรม 16500 us มาก",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("วิเคราะห์บัส I2C และขา INT")
    print("=" * 66)
    print(f"  อัตราสุ่ม  : {args.samplerate:,.0f} Hz")
    print(f"  ช่อง       : SCL=D{args.ch_scl}  SDA=D{args.ch_sda}  INT=D{args.ch_int}")
    print(f"  เกณฑ์แบ่งชุด: {args.gap_us:.0f} us")

    results: list[BusResult] = []
    for name in args.files:
        try:
            res = analyse_bus(
                Path(name),
                args.samplerate,
                args.ch_scl,
                args.ch_sda,
                args.ch_int,
                args.gap_us,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n  ข้ามไฟล์ {name}: {exc}")
            continue
        results.append(res)
        report(res)

    if not results:
        print("\nไม่มีไฟล์ใดวิเคราะห์ได้")
        sys.exit(1)

    compare(results)


if __name__ == "__main__":
    _main()

# ---------------------------------------------------------------------------
# ข้อจำกัดที่ทราบ
#
# 1. การแบ่งชุดการคุยใช้เกณฑ์ช่องว่างคงที่ 100 ไมโครวินาที
#    ถ้าไดรเวอร์เว้นช่วงกลางชุดนานกว่านั้น จะถูกนับเป็นสองชุด
#    ตรวจได้จากค่า พัลส์ต่อชุด ว่าสมเหตุสมผลหรือไม่
#    196 ไบต์ควรได้ประมาณ 196 คูณ 9 บวกส่วนหัว
#
# 2. เวลาที่บัสทำงาน วัดจากขอบ SCL แรกถึงขอบสุดท้ายของชุด
#    ไม่รวมช่วง start และ stop ซึ่งไม่มีพัลส์นาฬิกา
#    ค่าจริงจึงยาวกว่านี้เล็กน้อย ประมาณหนึ่งคาบนาฬิกาต่อชุด
#
# 3. ความถี่ SCL ใช้ค่ากลางของช่องว่างระหว่างขอบขาขึ้น
#    ช่วงที่ไดรเวอร์ยืดสัญญาณ หรือช่องว่างระหว่างไบต์ จะกลายเป็นค่าสุดโต่ง
#    ค่ากลางจึงเหมาะกว่าค่าเฉลี่ย
#
# 4. H5 วัดจากขอบขาลงของ INT ถึงพัลส์ SCL แรกที่ตามมา
#    ถ้าเซ็นเซอร์ยิง INT ระหว่างที่บัสยังคุยอยู่ ค่าที่ได้จะสั้นผิดปกติ
#    ดูได้จากค่าต่ำสุด ถ้าใกล้ศูนย์แปลว่ามีกรณีแบบนั้น
#
# 5. ยังไม่ได้ถอดรหัสเนื้อหาที่ส่ง ใช้การนับพัลส์ประมาณจำนวนไบต์เท่านั้น
#    ถ้าต้องการเนื้อหาจริง ใช้ sigrok-cli พร้อมตัวเลือก -P i2c
# ---------------------------------------------------------------------------
