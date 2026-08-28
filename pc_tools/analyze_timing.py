"""analyze_timing.py

อ่านไฟล์ log จาก PuTTY ที่บันทึกบรรทัด T ของเฟิร์มแวร์ ToF
แล้วสรุปสถิติเวลาต่อเฟรม + วาดกราฟสำหรับใส่ในรายงาน

รูปแบบบรรทัดที่รองรับ:
    H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us   (หัวตาราง)
    T,1,118,1928,55668,1444,32798,957                          (ข้อมูล)

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python analyze_timing.py                       # ใช้ไฟล์ .log ล่าสุดอัตโนมัติ
    python analyze_timing.py --file timing_before_blocking.log
    python analyze_timing.py --no-plot             # ไม่วาดกราฟ
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics as st
from typing import List, NamedTuple

# เฟรมแรก ๆ มีงาน setup ปนอยู่ ทำให้ค่าสูงผิดปกติ จึงตัดทิ้งก่อนคิดสถิติ
WARMUP_FRAMES = 2

# ความเร็วบัส I2C ที่ตั้งไว้ใน CubeMX (bit/s) ใช้คำนวณค่าตามทฤษฎีเพื่อเทียบ
I2C_CLOCK_HZ = 400_000

# ทุกไบต์บน I2C ใช้ 9 บิต (8 บิตข้อมูล + 1 บิต ACK)
BITS_PER_BYTE = 9


class Frame(NamedTuple):
    """ข้อมูลเวลา 1 เฟรม หน่วยเวลาเป็นไมโครวินาที"""

    frame: int
    rd_calls: int
    rd_bytes: int
    rd_us: int
    max_bytes: int
    max_us: int
    uart_us: int


def find_latest_log() -> str:
    """หาไฟล์ .log ที่แก้ไขล่าสุดในโฟลเดอร์ปัจจุบัน"""
    files = glob.glob("*.log")
    if not files:
        raise FileNotFoundError("ไม่พบไฟล์ .log ในโฟลเดอร์นี้")
    return max(files, key=os.path.getmtime)


def parse_log(path: str) -> List[Frame]:
    """อ่านไฟล์ log แล้วดึงเฉพาะบรรทัด T ที่มีครบ 8 ช่อง

    บรรทัดที่พังหรือขาด (เกิดจากกด reset กลางคัน) จะถูกข้าม ไม่ทำให้โปรแกรมตาย
    """
    frames: List[Frame] = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("T,"):
                continue
            parts = line.split(",")
            if len(parts) != 8:
                skipped += 1
                continue
            try:
                frames.append(Frame(*(int(p) for p in parts[1:])))
            except ValueError:
                skipped += 1

    if skipped:
        print(f"  (ข้ามบรรทัดที่อ่านไม่ได้ {skipped} บรรทัด)")
    return frames


def drop_warmup(frames: List[Frame]) -> List[Frame]:
    """ตัดเฟรม warm-up ออกจาก "ทุกรอบการรัน" ไม่ใช่แค่รอบแรก

    ไฟล์ log หนึ่งไฟล์อาจมีหลายรอบ ถ้ากดปุ่ม reset ระหว่างเก็บข้อมูล
    เลขเฟรมจะย้อนกลับไปเริ่มที่ 1 ใหม่ ซึ่งเฟรมแรกของทุกรอบมีงาน setup ปน
    ถ้าไม่ตัดออก ค่า sd จะพองผิดปกติ
    """
    kept: List[Frame] = []
    run_len = 0
    prev = -1
    runs = 1
    for f in frames:
        if f.frame <= prev:      # เลขเฟรมย้อนกลับ = เริ่มรอบใหม่
            run_len = 0
            runs += 1
        prev = f.frame
        run_len += 1
        if run_len > WARMUP_FRAMES:
            kept.append(f)
    if runs > 1:
        print(f"  (พบการรัน {runs} รอบในไฟล์นี้ ตัด warm-up ออกทุกรอบ)")
    return kept


def describe(name: str, values: List[float], unit: str = "") -> None:
    """พิมพ์สถิติบรรทัดเดียว: ค่าเฉลี่ย ส่วนเบี่ยงเบน ต่ำสุด สูงสุด"""
    sd = st.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {name:<22} {st.mean(values):>10.2f} {unit:<4}"
        f" sd {sd:>7.2f}   min {min(values):>9.2f}   max {max(values):>9.2f}"
    )


def report(frames: List[Frame], path: str) -> None:
    """พิมพ์รายงานสรุปทั้งหมด"""
    usable = drop_warmup(frames)
    if not usable:
        raise ValueError("ข้อมูลน้อยเกินไปหลังตัดเฟรม warm-up")

    print("=" * 72)
    print(f"ไฟล์: {path}")
    print(f"เฟรมทั้งหมด {len(frames)}  ใช้วิเคราะห์ {len(usable)} "
          f"(ตัด warm-up {WARMUP_FRAMES} เฟรมแรกของทุกรอบ)")
    print("=" * 72)

    rd_ms = [f.rd_us / 1000 for f in usable]
    mx_ms = [f.max_us / 1000 for f in usable]
    ua_ms = [f.uart_us / 1000 for f in usable]
    total_ms = [(f.rd_us + f.uart_us) / 1000 for f in usable]

    print("\n[1] เวลาต่อเฟรม")
    describe("I2C รวม", rd_ms, "ms")
    describe("  - โอนข้อมูลก้อนใหญ่", mx_ms, "ms")
    describe("  - poll สถานะ", [a - b for a, b in zip(rd_ms, mx_ms)], "ms")
    describe("UART", ua_ms, "ms")
    describe("รวมทั้งหมด", total_ms, "ms")

    print("\n[2] ปริมาณข้อมูล")
    describe("เรียก RdMulti", [float(f.rd_calls) for f in usable], "ครั้ง")
    describe("ไบต์รวม", [float(f.rd_bytes) for f in usable], "B")
    describe("ก้อนใหญ่สุด", [float(f.max_bytes) for f in usable], "B")

    # เทียบกับทฤษฎี เพื่อดูว่าโค้ดเรามี overhead หรือไม่
    big_b = st.mean(f.max_bytes for f in usable)
    theo_ms = (big_b + 4) * BITS_PER_BYTE / I2C_CLOCK_HZ * 1000
    meas_ms = st.mean(mx_ms)
    print("\n[3] เทียบกับทฤษฎี (การโอนข้อมูลก้อนใหญ่)")
    print(f"  ทฤษฎีที่ {I2C_CLOCK_HZ/1000:.0f} kHz : {theo_ms:8.2f} ms")
    print(f"  วัดจริง                : {meas_ms:8.2f} ms")
    print(f"  ประสิทธิภาพบัส         : {theo_ms/meas_ms*100:8.1f} %")

    # สัดส่วนเวลาที่ใช้ เทียบกับงบเวลาต่อเฟรม
    poll_ms = meas_ms and (st.mean(rd_ms) - meas_ms)
    print("\n[4] เวลาไปไหนบ้าง")
    tot = st.mean(total_ms)
    for label, val in (
        ("โอนข้อมูล", meas_ms),
        ("poll สถานะ", poll_ms),
        ("UART", st.mean(ua_ms)),
    ):
        bar = "#" * int(val / tot * 40)
        print(f"  {label:<12} {val:7.2f} ms  {val/tot*100:5.1f}%  {bar}")
    print(f"  {'รวม':<12} {tot:7.2f} ms")


def make_plot(frames: List[Frame], out_png: str) -> None:
    """วาดกราฟเวลาต่อเฟรม บันทึกเป็น PNG สำหรับใส่รายงาน"""
    import matplotlib
    matplotlib.use("Agg")            # ไม่ต้องเปิดหน้าต่าง เซฟไฟล์อย่างเดียว
    import matplotlib.pyplot as plt

    usable = drop_warmup(frames)
    x = list(range(1, len(usable) + 1))   # ลำดับตัวอย่าง กันเลขเฟรมซ้ำข้ามรอบ
    big = [f.max_us / 1000 for f in usable]
    poll = [(f.rd_us - f.max_us) / 1000 for f in usable]
    uart = [f.uart_us / 1000 for f in usable]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(
        x, big, poll, uart,
        labels=["I2C data transfer", "I2C status polling", "UART transmit"],
        alpha=0.85,
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Time per frame (ms)")
    ax.set_title("Per-frame latency breakdown (blocking mode, I2C @ 400 kHz)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nบันทึกกราฟแล้ว: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", help="ไฟล์ log (ถ้าไม่ระบุ จะใช้ .log ล่าสุด)")
    ap.add_argument("--no-plot", action="store_true", help="ไม่ต้องวาดกราฟ")
    args = ap.parse_args()

    path = args.file or find_latest_log()
    frames = parse_log(path)
    if not frames:
        raise SystemExit("ไม่พบบรรทัด T ในไฟล์นี้ — ตรวจว่าเฟิร์มแวร์อยู่ในโหมดวัดเวลา")

    report(frames, path)

    if not args.no_plot:
        out = os.path.splitext(path)[0] + ".png"
        make_plot(frames, out)

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
