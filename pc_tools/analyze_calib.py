"""
analyze_calib.py
================
วัดความถี่ของคลื่นสอบเทียบจากไฟล์ที่ sigrok-cli บันทึกไว้

ทำไมต้องมีไฟล์นี้
------------------
PROJECT_FACTS TRAP หมายเลข 12 ระบุว่าทั้งโหมด 8x8 และ 4x4 เร็วเกินสเปก
0.7 เปอร์เซ็นต์เท่ากัน ผู้ต้องสงสัยคือ RC oscillator ในเซ็นเซอร์ กับนาฬิกาบอร์ด
ตอนนี้มีนาฬิกาสองตัวที่เถียงกัน จึงตัดสินไม่ได้

logic analyzer มีคริสตัลของตัวเอง เป็นนาฬิกาตัวที่สามที่เป็นอิสระ
แต่ก่อนใช้มันตัดสิน ต้องรู้ก่อนว่ามันเองแม่นแค่ไหน
จึงให้บอร์ดปล่อยคลื่นที่รู้ค่าแน่นอน แล้วให้ logic analyzer วัด

ทำไมต้องวัดหลายพันลูก ไม่ใช่ลูกเดียว
------------------------------------
ที่ sample rate 1 MHz ความละเอียดเวลาคือ 1 ไมโครวินาที
วัดคลื่นลูกเดียว คาบ 1000 ไมโครวินาที คลาดได้ถึง 1000 ppm
เราต้องแยกความต่าง 0.7 เปอร์เซ็นต์ ซึ่งเท่ากับ 7000 ppm
1000 เทียบกับ 7000 ใกล้กันเกินไป สรุปไม่ได้

วัดจากขอบแรกถึงขอบสุดท้าย แล้วหารด้วยจำนวนคาบ
ความคลาดยังเท่าเดิมคือ 1 ไมโครวินาที แต่หารด้วยเวลารวมที่ยาวมาก
10 วินาที ให้ความคลาดเหลือ 0.1 ppm ซึ่งละเอียดกว่าที่ต้องการ 70000 เท่า

วิธีรัน
-------
    python analyze_calib.py cal1k_run1.bin
    python analyze_calib.py cal1k_run1.bin cal1k_run2.bin cal1k_run3.bin
    python analyze_calib.py cal1k_run1.bin --expected 1000.0 --samplerate 1000000
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


# จำนวนตัวอย่างที่ตัดทิ้งหัวและท้าย
#
# เหตุผล ไฟล์ที่ sigrok-cli สร้างมีส่วนหัวติดมาด้วย
# (สังเกตจากขนาดไฟล์ 10,000,022 ไบต์ ขณะที่ขอ 10,000,000 ตัวอย่าง)
# ถ้าตีความส่วนหัวเป็นข้อมูล จะเกิดขอบสัญญาณปลอมตอนต้นไฟล์
#
# แทนที่จะเดาว่าส่วนหัวยาวเท่าไหร่ เราตัดทิ้งหัวและท้ายไปเลยอย่างละมาก ๆ
# แล้ววิเคราะห์เฉพาะช่วงกลาง วิธีนี้ถูกต้องโดยไม่ต้องรู้รูปแบบไฟล์
# และเสียข้อมูลไปเพียง 0.02 เปอร์เซ็นต์ ซึ่งไม่มีผลต่อความแม่น
GUARD_SAMPLES: int = 1000

# เครื่องหมายที่ sigrok-cli ใส่ไว้หัวและท้ายไฟล์เมื่อใช้รูปแบบ binary
#
# ค้นพบ 9 ก.ย. 2026 จากการเปิดไฟล์ด้วยโปรแกรมแก้ข้อความ
# นับความยาวได้ 12 บวก 10 เท่ากับ 22 ไบต์ ตรงกับส่วนที่เกินมาพอดี
# จากไฟล์ที่ขอ 10,000,000 ตัวอย่าง แต่ได้ขนาด 10,000,022 ไบต์
#
# ตัวอักษรเหล่านี้ถ้าถูกตีความเป็นข้อมูลสัญญาณ จะกลายเป็นขอบปลอม
# ตัวอักษร F มีรหัส 0x46 ซึ่งบิตล่างสุดเป็น 0
# ตัวอักษร R มีรหัส 0x52 ซึ่งบิตล่างสุดเป็น 0
# ตัวอักษร A มีรหัส 0x41 ซึ่งบิตล่างสุดเป็น 1
# จึงเกิดการเปลี่ยนสถานะปลอมได้จริง ต้องตัดออกก่อนวิเคราะห์
MARKER_BEGIN: bytes = b"FRAME-BEGIN\n"
MARKER_END: bytes = b"FRAME-END\n"


def strip_markers(raw: np.ndarray, verbose: bool = True) -> np.ndarray:
    """ตัดเครื่องหมาย FRAME-BEGIN และ FRAME-END ออกจากข้อมูลดิบ

    ถ้าไม่พบเครื่องหมาย จะคืนข้อมูลเดิมโดยไม่ตัดอะไร
    แล้วให้ GUARD_SAMPLES ทำหน้าที่ป้องกันแทน

    ทำไมต้องตัดตรง ๆ แทนที่จะตัดหัวท้ายทิ้งเฉย ๆ
    ---------------------------------------------
    การตัดทิ้งแบบเผื่อ ๆ ได้ผลถูกต้องก็จริง แต่เป็นความบังเอิญ
    ไม่ใช่การออกแบบ ถ้าวันหน้า sigrok เปลี่ยนความยาวเครื่องหมาย
    หรือเราเปลี่ยนค่าเผื่อ ก็อาจพลาดโดยไม่รู้ตัว
    การตัดตามโครงสร้างจริง ทำให้ตรวจสอบได้ว่าเหลือข้อมูลครบตามที่ขอหรือไม่

    Args:
        raw: ข้อมูลดิบทั้งไฟล์
        verbose: พิมพ์รายงานว่าพบเครื่องหมายหรือไม่

    Returns:
        ข้อมูลที่ตัดเครื่องหมายออกแล้ว
    """
    data = raw
    found_begin = False
    found_end = False

    if data.size >= len(MARKER_BEGIN):
        head = data[: len(MARKER_BEGIN)].tobytes()
        if head == MARKER_BEGIN:
            data = data[len(MARKER_BEGIN) :]
            found_begin = True

    if data.size >= len(MARKER_END):
        tail = data[-len(MARKER_END) :].tobytes()
        if tail == MARKER_END:
            data = data[: -len(MARKER_END)]
            found_end = True

    if verbose:
        if found_begin and found_end:
            print(
                f"  เครื่องหมายไฟล์     : พบครบ ตัดออก "
                f"{len(MARKER_BEGIN) + len(MARKER_END)} ไบต์"
            )
            print(f"  ตัวอย่างที่เหลือจริง : {data.size:,}")
        elif found_begin or found_end:
            print("  เตือน: พบเครื่องหมายไม่ครบทั้งหัวและท้าย")
            print("  ไฟล์อาจถูกตัดกลางคัน ควรเก็บข้อมูลใหม่")
        else:
            print("  เครื่องหมายไฟล์     : ไม่พบ ใช้วิธีตัดหัวท้ายเผื่อแทน")

    return data


@dataclass
class CalibResult:
    """ผลการวิเคราะห์ไฟล์หนึ่งไฟล์"""

    path: str
    n_samples: int          # จำนวนตัวอย่างที่นำมาวิเคราะห์จริง
    n_periods: int          # จำนวนคาบที่วัดได้
    span_samples: int       # ระยะจากขอบขาขึ้นแรกถึงขอบขาขึ้นสุดท้าย
    mean_period: float      # คาบเฉลี่ย หน่วยตัวอย่าง
    freq_hz: float          # ความถี่ที่วัดได้
    error_ppm: float        # ความคลาดเทียบกับค่าที่บอร์ดรายงาน
    duty_percent: float     # สัดส่วนเวลาที่สัญญาณอยู่สถานะสูง
    period_min: int         # คาบสั้นที่สุดที่พบ
    period_max: int         # คาบยาวที่สุดที่พบ
    period_std: float       # ส่วนเบี่ยงเบนมาตรฐานของคาบ
    n_outliers: int         # จำนวนคาบที่ผิดปกติ ดูเกณฑ์ในฟังก์ชัน analyse


def show_header(path: Path, n_bytes: int = 32) -> None:
    """พิมพ์ไบต์แรกของไฟล์ให้ดู เพื่อทราบว่าส่วนหัวหน้าตาอย่างไร

    ไม่ได้ใช้ในการคำนวณ แต่ช่วยให้เข้าใจรูปแบบไฟล์
    ซึ่งเป็นคำถามที่ยังไม่มีคำตอบตั้งแต่ตอนทดสอบครั้งแรก

    Args:
        path: ไฟล์ที่ต้องการดู
        n_bytes: จำนวนไบต์ที่จะแสดง
    """
    with open(path, "rb") as f:
        head = f.read(n_bytes)

    hex_part = " ".join(f"{b:02X}" for b in head)
    txt_part = "".join(chr(b) if 32 <= b < 127 else "." for b in head)

    print(f"  ไบต์แรก (hex) : {hex_part}")
    print(f"  ไบต์แรก (text): {txt_part}")


def analyse(
    path: Path,
    samplerate: float,
    expected_hz: float,
    channel_bit: int,
    outlier_tolerance: float,
) -> CalibResult:
    """วิเคราะห์ไฟล์หนึ่งไฟล์

    Args:
        path: ไฟล์ binary จาก sigrok-cli
        samplerate: อัตราสุ่มที่ตั้งไว้ หน่วย Hz
        expected_hz: ความถี่ที่บอร์ดรายงานว่าปล่อยออกมา
        channel_bit: บิตที่เก็บช่องสัญญาณ D0 คือบิต 0
        outlier_tolerance: คาบที่ต่างจากค่ากลางเกินสัดส่วนนี้ นับเป็นค่าผิดปกติ

    Returns:
        CalibResult

    Raises:
        FileNotFoundError: หาไฟล์ไม่เจอ
        ValueError: ข้อมูลน้อยเกินไป หรือไม่พบขอบสัญญาณ
    """
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    raw = np.fromfile(path, dtype=np.uint8)

    if raw.size < 4 * GUARD_SAMPLES:
        raise ValueError(f"ข้อมูลน้อยเกินไป มีเพียง {raw.size} ไบต์")

    # ตัดหัวและท้ายทิ้ง เพื่อเลี่ยงส่วนหัวไฟล์และขอบท้ายที่อาจไม่สมบูรณ์
    core = raw[GUARD_SAMPLES:-GUARD_SAMPLES]

    # ดึงเฉพาะบิตของช่องที่สนใจ ได้อาร์เรย์ที่มีแต่ 0 กับ 1
    bits = (core >> channel_bit) & 1

    # หาขอบขาขึ้น คือตำแหน่งที่ค่าเปลี่ยนจาก 0 เป็น 1
    diff = np.diff(bits.astype(np.int8))
    rising = np.flatnonzero(diff == 1) + 1

    if rising.size < 3:
        raise ValueError(
            f"พบขอบขาขึ้นเพียง {rising.size} จุด ไม่พอวิเคราะห์ "
            "ตรวจว่าต่อสายถูกช่องหรือไม่ และบอร์ดกำลังปล่อยคลื่นอยู่หรือไม่"
        )

    # หัวใจของการวัด
    # ไม่ได้เฉลี่ยคาบทีละลูก แต่วัดระยะรวมจากขอบแรกถึงขอบสุดท้าย
    # แล้วหารด้วยจำนวนคาบ
    #
    # ความคลาดจากการสุ่มเกิดที่ปลายทั้งสองข้างเท่านั้น คือประมาณ 1 ตัวอย่าง
    # ยิ่งช่วงยาว ความคลาดต่อคาบยิ่งเล็กลง
    span = int(rising[-1] - rising[0])
    n_periods = int(rising.size - 1)
    mean_period = span / n_periods

    freq = samplerate / mean_period
    error_ppm = (freq - expected_hz) / expected_hz * 1e6

    # สถิติของคาบรายลูก ใช้ตรวจว่ามีข้อมูลตกหล่นหรือไม่
    periods = np.diff(rising)
    median_period = float(np.median(periods))
    tol = median_period * outlier_tolerance
    n_outliers = int(np.count_nonzero(np.abs(periods - median_period) > tol))

    duty = float(np.count_nonzero(bits)) / bits.size * 100.0

    return CalibResult(
        path=str(path),
        n_samples=int(core.size),
        n_periods=n_periods,
        span_samples=span,
        mean_period=mean_period,
        freq_hz=freq,
        error_ppm=error_ppm,
        duty_percent=duty,
        period_min=int(periods.min()),
        period_max=int(periods.max()),
        period_std=float(periods.std()),
        n_outliers=n_outliers,
    )


def report(res: CalibResult, expected_hz: float, samplerate: float) -> None:
    """พิมพ์ผลการวิเคราะห์ไฟล์เดียว"""
    print(f"\n--- {Path(res.path).name} ---")
    print(f"  ตัวอย่างที่ใช้     : {res.n_samples:,}")
    print(f"  จำนวนคาบที่วัดได้  : {res.n_periods:,}")
    print(f"  ระยะรวมที่ใช้วัด   : {res.span_samples:,} ตัวอย่าง")
    print()
    print(f"  คาบเฉลี่ย          : {res.mean_period:.6f} ตัวอย่าง")
    print(f"  ความถี่ที่วัดได้    : {res.freq_hz:.6f} Hz")
    print(f"  บอร์ดรายงานว่า     : {expected_hz:.6f} Hz")
    print(f"  ความคลาด           : {res.error_ppm:+.3f} ppm")
    print()
    print(f"  duty cycle         : {res.duty_percent:.3f} %")
    print(f"  คาบสั้นสุด/ยาวสุด   : {res.period_min} / {res.period_max} ตัวอย่าง")
    print(f"  ส่วนเบี่ยงเบนคาบ    : {res.period_std:.3f} ตัวอย่าง")

    # ความละเอียดของการวัดครั้งนี้
    # ความคลาดที่ปลายทั้งสองข้างประมาณ 1 ตัวอย่าง หารด้วยระยะรวม
    resolution_ppm = 1.0 / res.span_samples * 1e6
    print(f"  ความละเอียดที่ทำได้ : {resolution_ppm:.4f} ppm")

    if res.n_outliers > 0:
        print()
        print(f"  เตือน: พบคาบผิดปกติ {res.n_outliers:,} ลูก")
        print("  สาเหตุที่เป็นไปได้ ข้อมูลตกหล่นระหว่างส่งผ่าน USB")
        print("  หรือมีสัญญาณรบกวนทำให้เกิดขอบปลอม")
        print("  ควรเก็บข้อมูลใหม่ที่อัตราสุ่มต่ำลง แล้วเทียบกัน")


def compare(results: list[CalibResult]) -> None:
    """เทียบผลหลายรอบ เพื่อดูว่าทำซ้ำได้หรือไม่

    กฎที่ตั้งไว้ในโปรเจคคือ n น้อยกว่า 3 ห้ามสรุป
    ฟังก์ชันนี้จึงเตือนเมื่อมีข้อมูลไม่ถึงสามรอบ
    """
    print("\n" + "=" * 60)
    print("เทียบผลระหว่างรอบ")
    print("=" * 60)

    if len(results) < 3:
        print(f"  เตือน: มีเพียง {len(results)} รอบ")
        print("  ตามกฎของโปรเจค ต้องมีอย่างน้อย 3 รอบจึงจะสรุปได้")
        print("  เก็บข้อมูลเพิ่มแล้วรันใหม่พร้อมกันทุกไฟล์")

    freqs = np.array([r.freq_hz for r in results])
    ppms = np.array([r.error_ppm for r in results])

    print(f"\n  {'ไฟล์':<24} {'ความถี่ (Hz)':>16} {'คลาด (ppm)':>14}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {Path(r.path).name:<24} {r.freq_hz:>16.6f} {r.error_ppm:>+14.3f}")

    if len(results) >= 2:
        spread = float(freqs.max() - freqs.min())
        spread_ppm = float(ppms.max() - ppms.min())
        print("  " + "-" * 56)
        print(f"  {'ค่าเฉลี่ย':<24} {freqs.mean():>16.6f} {ppms.mean():>+14.3f}")
        print(f"  {'ส่วนเบี่ยงเบน':<24} {freqs.std():>16.6f} {ppms.std():>14.3f}")
        print(f"  {'ช่วงกว้างสุด':<24} {spread:>16.6f} {spread_ppm:>14.3f}")


def interpret(mean_ppm: float, threshold_ppm: float) -> None:
    """แปลผลว่าตอบสมมติฐาน H1 ได้อย่างไร"""
    print("\n" + "=" * 60)
    print("การแปลผล")
    print("=" * 60)

    print(f"\n  ความคลาดเฉลี่ยที่วัดได้: {mean_ppm:+.3f} ppm")
    print(f"  ค่าที่ต้องแยกให้ออกใน TRAP หมายเลข 12: 7000 ppm ซึ่งคือ 0.7 เปอร์เซ็นต์")
    print()

    if abs(mean_ppm) < threshold_ppm:
        print("  ผลสรุป นาฬิกาของบอร์ดกับของ logic analyzer ตรงกัน")
        print()
        print("  นาฬิกาสองตัวนี้เป็นอิสระต่อกัน คือคริสตัลคนละตัว คนละวงจร")
        print("  โอกาสที่ทั้งคู่จะเพี้ยนไปในทางเดียวกันด้วยขนาดเท่ากันพอดี")
        print("  แทบเป็นศูนย์ จึงสรุปได้ว่าทั้งคู่ถูกต้อง")
        print()
        print("  ดังนั้น logic analyzer ใช้เป็นนาฬิกาอ้างอิงที่สามได้")
        print("  ขั้นถัดไปคือวัดคาบเฟรมของเซ็นเซอร์จริง")
        print("  ถ้าวัดได้ตรงกับที่ DWT บนบอร์ดวัดไว้ แปลว่าเซ็นเซอร์คือตัวที่เร็ว")
    else:
        print("  ผลสรุป นาฬิกาสองตัวไม่ตรงกัน")
        print()
        print("  ยังตัดสิน TRAP หมายเลข 12 ไม่ได้ ต้องหาสาเหตุก่อนว่า")
        print("  ตัวไหนเพี้ยน ระหว่างนาฬิกาบอร์ดกับนาฬิกา logic analyzer")
        print()
        print("  ข้อควรตรวจ")
        print("    - ค่า MY_CAL_FREQ_HZ ในเฟิร์มแวร์ ตรงกับที่ใส่ในตัวเลือก expected หรือไม่")
        print("    - อัตราสุ่มที่ตั้งใน sigrok-cli ตรงกับตัวเลือก samplerate หรือไม่")
        print("    - มีคาบผิดปกติในรายงานข้างบนหรือไม่")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="วัดความถี่คลื่นสอบเทียบจากไฟล์ binary ของ sigrok-cli"
    )
    parser.add_argument("files", nargs="+", help="ไฟล์ .bin ที่ต้องการวิเคราะห์")
    parser.add_argument(
        "--samplerate",
        type=float,
        default=1_000_000.0,
        help="อัตราสุ่มที่ตั้งไว้ตอนเก็บข้อมูล หน่วย Hz ค่าเริ่มต้น 1000000",
    )
    parser.add_argument(
        "--expected",
        type=float,
        default=1000.0,
        help="ความถี่ที่บอร์ดรายงาน หน่วย Hz ค่าเริ่มต้น 1000",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="หมายเลขช่อง D0 คือ 0 ค่าเริ่มต้น 0",
    )
    parser.add_argument(
        "--threshold-ppm",
        type=float,
        default=100.0,
        help="เกณฑ์ตัดสินว่านาฬิกาตรงกัน หน่วย ppm ค่าเริ่มต้น 100",
    )
    parser.add_argument(
        "--outlier-tolerance",
        type=float,
        default=0.10,
        help="คาบที่ต่างจากค่ากลางเกินสัดส่วนนี้ นับว่าผิดปกติ ค่าเริ่มต้น 0.10",
    )
    parser.add_argument(
        "--show-header",
        action="store_true",
        help="แสดงไบต์แรกของไฟล์ เพื่อดูรูปแบบส่วนหัว",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("วิเคราะห์คลื่นสอบเทียบ")
    print("=" * 60)
    print(f"  อัตราสุ่มที่ตั้งไว้ : {args.samplerate:,.0f} Hz")
    print(f"  บอร์ดรายงานว่าปล่อย : {args.expected:.6f} Hz")
    print(f"  ช่องที่อ่าน        : D{args.channel}")

    results: list[CalibResult] = []
    for name in args.files:
        path = Path(name)
        try:
            if args.show_header:
                print(f"\n--- ส่วนหัวของ {path.name} ---")
                show_header(path)

            res = analyse(
                path,
                args.samplerate,
                args.expected,
                args.channel,
                args.outlier_tolerance,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n  ข้ามไฟล์ {name}: {exc}")
            continue

        results.append(res)
        report(res, args.expected, args.samplerate)

    if not results:
        print("\nไม่มีไฟล์ใดวิเคราะห์ได้")
        sys.exit(1)

    compare(results)

    mean_ppm = float(np.mean([r.error_ppm for r in results]))
    interpret(mean_ppm, args.threshold_ppm)


if __name__ == "__main__":
    _main()

# ---------------------------------------------------------------------------
# ข้อจำกัดที่ทราบ
#
# 1. สคริปต์ตัดหัวและท้ายทิ้งอย่างละ 1000 ตัวอย่าง แทนการอ่านส่วนหัวไฟล์
#    วิธีนี้ถูกต้องโดยไม่ต้องรู้รูปแบบไฟล์ แต่ถ้าส่วนหัวยาวเกิน 1000 ไบต์
#    จะยังมีข้อมูลปลอมปนอยู่ ปัจจุบันส่วนหัวยาว 22 ไบต์ จึงเผื่อไว้มากพอ
#
# 2. ความถี่ที่วัดได้เป็นอัตราส่วนระหว่างนาฬิกาบอร์ดกับนาฬิกา logic analyzer
#    ไม่ใช่ค่าสัมบูรณ์ ถ้าทั้งคู่เพี้ยนเท่ากันพอดี สคริปต์นี้จับไม่ได้
#    แต่โอกาสเกิดแทบเป็นศูนย์ เพราะเป็นคริสตัลคนละตัวคนละวงจร
#
# 3. duty cycle ที่คำนวณ นับจากสัดส่วนตัวอย่างที่เป็นสถานะสูงทั้งไฟล์
#    ถ้าไฟล์เริ่มหรือจบกลางคลื่น จะคลาดเล็กน้อย ไม่มีผลต่อการวัดความถี่
#
# 4. ยังไม่ได้ตรวจสอบว่าไฟล์บันทึกครบตามจำนวนที่ขอหรือไม่
#    ต้องดูขนาดไฟล์เองก่อนรัน
#
# 5. เกณฑ์ตัดสิน 100 ppm เป็นค่าที่ตั้งขึ้นเอง โดยดูจากสองสิ่ง
#    คริสตัลของ logic analyzer วัดได้ประมาณ 79 ppm ตามที่บันทึกใน PROJECT_FACTS
#    และค่าที่ต้องแยกให้ออกคือ 7000 ppm ซึ่งห่างกันมาก
#    ถ้าผลออกมาอยู่ระหว่าง 100 ถึง 7000 ppm ต้องพิจารณาเป็นกรณีไป
# ---------------------------------------------------------------------------
