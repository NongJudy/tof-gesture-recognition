"""
drift_run.py
============
ทดลองดริฟต์ของอัตราเฟรม เก็บข้อมูลอัตโนมัติ แล้ววิเคราะห์

ทำไมต้องอัตโนมัติ
-----------------
การทดลองนี้ต้องวัดซ้ำทุกไม่กี่นาทีเป็นเวลาหนึ่งชั่วโมงครึ่ง
ถ้าให้คนนั่งพิมพ์คำสั่งเอง จะเกิดปัญหาสามอย่าง
  1. ช่วงเวลาระหว่างการวัดไม่สม่ำเสมอ ทำให้เส้นโค้งบิดเบือน
  2. เวลาที่บันทึกเป็นค่าประมาณ ไม่ใช่เวลาจริง
  3. พลาดง่าย เช่น ลืมรอบ หรือตั้งชื่อไฟล์ผิด

ปัญหาข้อ 1 และ 2 เป็นสาเหตุที่กราฟ fig5 ชุดแรกต้องติดป้ายว่าเป็นผลเบื้องต้น
สคริปต์นี้แก้ทั้งสามข้อ โดยบันทึกเวลาจริงลงไฟล์ manifest ทุกรอบ

วิธีรัน
-------
ขั้นเก็บข้อมูล ต้องปิด PulseView และ PuTTY ก่อน

    python drift_run.py capture --outdir ..\\data\\phase1_drift

ขั้นวิเคราะห์ รันเมื่อไหร่ก็ได้หลังเก็บเสร็จ

    python drift_run.py analyse --outdir ..\\data\\phase1_drift
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ต้องติดตั้ง numpy ก่อน  พิมพ์:  pip install numpy")
    sys.exit(1)


SIGROK = r"C:\Program Files\sigrok\sigrok-cli\sigrok-cli.exe"
GUARD_SAMPLES: int = 1000
CAL_PPM: float = -95.408          # ค่าสอบเทียบ logic analyzer เทียบนาฬิกาบอร์ด


def fit_period(edges: np.ndarray) -> tuple[float, float, float]:
    """หาคาบเฉลี่ยด้วยการปรับเส้นตรงกับตำแหน่งขอบทุกอัน

    สำเนาจาก analyze_calib.py เพื่อให้ไฟล์นี้รันได้ด้วยตัวเอง
    และเพื่อให้ตัวเลขทุกที่ในโปรเจคคำนวณด้วยวิธีเดียวกัน

    Args:
        edges: ตำแหน่งขอบสัญญาณ หน่วยตัวอย่าง

    Returns:
        (คาบเฉลี่ย, ความไม่แน่นอน, รากที่สองของกำลังสองเฉลี่ยของส่วนตกค้าง)
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
    rv = float(np.sum(resid * resid)) / (n - 2)
    return (b, float(np.sqrt(rv / sxx)), float(np.sqrt(rv)))


def capture_one(out: Path, samples: int, samplerate: str) -> bool:
    """เรียก sigrok-cli เก็บข้อมูลหนึ่งครั้ง

    Args:
        out: ไฟล์ปลายทาง
        samples: จำนวนตัวอย่างที่ขอ
        samplerate: อัตราสุ่ม เช่น 1m

    Returns:
        True ถ้าได้ไฟล์ที่มีขนาดใกล้เคียงที่ขอ
    """
    cmd = [SIGROK, "-d", "fx2lafw", "--config", f"samplerate={samplerate}",
           "--samples", str(samples), "-O", "binary", "-o", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"    เรียก sigrok-cli ไม่สำเร็จ: {exc}")
        return False

    if not out.is_file():
        print(f"    ไม่ได้ไฟล์  stderr: {r.stderr.strip()[:120]}")
        return False

    size = out.stat().st_size
    # ยอมรับส่วนเกินได้ไม่เกิน 100 ไบต์ ปกติเกิน 22 ไบต์จากส่วนหัวและท้ายไฟล์
    if abs(size - samples) > 100:
        print(f"    ขนาดไฟล์ผิดปกติ {size:,} ไบต์ ขอไว้ {samples:,} — อาจมีข้อมูลตก")
        return False
    return True


def cmd_capture(args: argparse.Namespace) -> None:
    """เก็บข้อมูลเป็นรอบ ๆ ตามช่วงเวลาที่กำหนด"""
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"

    n_total = int(args.duration_min * 60 / args.interval_s) + 1

    print("=" * 62)
    print("ทดลองดริฟต์ — ขั้นเก็บข้อมูล")
    print("=" * 62)
    print(f"  ช่วงห่างระหว่างรอบ : {args.interval_s} วินาที")
    print(f"  ระยะเวลารวม       : {args.duration_min} นาที")
    print(f"  จำนวนรอบ          : {n_total}")
    print(f"  แต่ละรอบเก็บ       : {args.samples:,} ตัวอย่างที่ {args.samplerate}")
    print(f"  เนื้อที่ที่ต้องใช้   : ประมาณ {n_total * args.samples / 1e6:.0f} MB")
    print(f"  เก็บไว้ที่          : {out}")
    print()
    print("  ห้ามแตะสาย ห้ามขยับบอร์ด ตลอดการทดลอง")
    print("  ปิด PulseView และ PuTTY ให้สนิทก่อน")
    print()

    t0 = time.time()
    t0_wall = datetime.now()

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "elapsed_s", "wall_clock", "filename", "ok"])

        for i in range(n_total):
            target = t0 + i * args.interval_s
            wait = target - time.time()
            if wait > 0:
                time.sleep(wait)

            elapsed = time.time() - t0
            wall = datetime.now()
            name = f"drift_{i:03d}.bin"

            print(f"  [{i + 1:3d}/{n_total}] t = {elapsed / 60:6.2f} นาที  "
                  f"{wall.strftime('%H:%M:%S')}  ", end="", flush=True)

            ok = capture_one(out / name, args.samples, args.samplerate)
            print("เก็บแล้ว" if ok else "ล้มเหลว")

            w.writerow([i, f"{elapsed:.1f}", wall.isoformat(timespec="seconds"),
                        name, int(ok)])
            f.flush()          # เขียนลงดิสก์ทันที เผื่อไฟดับกลางทาง

    print()
    print(f"เสร็จ  เริ่ม {t0_wall.strftime('%H:%M:%S')}  "
          f"จบ {datetime.now().strftime('%H:%M:%S')}")
    print(f"บันทึกเวลาไว้ที่ {manifest}")


def cmd_analyse(args: argparse.Namespace) -> None:
    """วิเคราะห์ไฟล์ทั้งหมดตาม manifest แล้วสร้างตารางและกราฟ"""
    out = Path(args.outdir)
    manifest = out / "manifest.csv"
    if not manifest.is_file():
        print(f"ไม่พบ {manifest}  ต้องรันขั้น capture ก่อน")
        sys.exit(1)

    rows = []
    with open(manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ok"] == "1":
                rows.append(r)

    print("=" * 62)
    print("ทดลองดริฟต์ — ขั้นวิเคราะห์")
    print("=" * 62)
    print(f"  ไฟล์ที่จะวิเคราะห์ : {len(rows)}")
    print()

    results = []
    for r in rows:
        path = out / r["filename"]
        if not path.is_file():
            print(f"  ข้าม {r['filename']} — หาไฟล์ไม่เจอ")
            continue

        raw = np.fromfile(path, dtype=np.uint8)
        core = raw[GUARD_SAMPLES:-GUARD_SAMPLES]
        bits = (core >> args.channel) & 1
        d = np.diff(bits.astype(np.int8))
        falling = np.flatnonzero(d == -1) + 1

        if falling.size < 10:
            print(f"  ข้าม {r['filename']} — พบขอบเพียง {falling.size} จุด")
            continue

        per, se, jit = fit_period(falling)
        freq_raw = args.samplerate_hz / per
        freq = freq_raw * (1 - CAL_PPM / 1e6)        # แก้ตามค่าสอบเทียบ
        ppm = (freq / args.expected - 1) * 1e6
        se_ppm = se / per * 1e6

        results.append({
            "elapsed_min": float(r["elapsed_s"]) / 60.0,
            "wall": r["wall_clock"],
            "file": r["filename"],
            "n_periods": int(falling.size - 1),
            "period_ms": per / args.samplerate_hz * 1000.0,
            "freq_hz": freq,
            "ppm": ppm,
            "se_ppm": se_ppm,
            "jitter_us": jit / args.samplerate_hz * 1e6,
        })
        print(f"  {r['filename']}  t={results[-1]['elapsed_min']:6.2f} นาที  "
              f"{ppm:+9.1f} ppm  ±{se_ppm:.1f}")

    if not results:
        print("ไม่มีไฟล์ใดวิเคราะห์ได้")
        sys.exit(1)

    csv_out = out / "drift_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  ตารางผล -> {csv_out.name}")

    t = np.array([r["elapsed_min"] for r in results])
    y = np.array([r["ppm"] for r in results])

    print()
    print("  สรุป")
    print(f"    ค่าแรก        : {y[0]:+.1f} ppm  ที่ t = {t[0]:.2f} นาที")
    print(f"    ค่าสุดท้าย     : {y[-1]:+.1f} ppm  ที่ t = {t[-1]:.2f} นาที")
    print(f"    เปลี่ยนไปทั้งหมด : {y[-1] - y[0]:+.1f} ppm")
    print(f"    ต่ำสุด/สูงสุด   : {y.min():+.1f} / {y.max():+.1f} ppm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  ไม่มี matplotlib จึงไม่ได้สร้างกราฟ  พิมพ์: pip install matplotlib")
        return

    e = np.array([r["se_ppm"] for r in results])
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.errorbar(t, y, yerr=e, fmt="o-", color="tab:blue", markersize=4,
                linewidth=1.2, capsize=2, ecolor="tab:blue", alpha=0.9)
    ax.set_xlabel("Time since power-on (minutes)")
    ax.set_ylabel(f"Frame rate deviation from {args.expected:.0f} Hz (ppm)")
    ax.set_title("Sensor frame rate after cold power-on")
    ax.grid(alpha=0.3)
    ax.text(0.99, 0.02,
            f"n = {len(results)} captures, {args.duration_note}\n"
            f"Least-squares fit over all edges; calibration applied.\n"
            f"Error bars are the fit standard error.",
            transform=ax.transAxes, fontsize=7, color="dimgray",
            va="bottom", ha="right")
    fig.tight_layout()
    fig_out = out / "fig9_drift_coldstart.png"
    fig.savefig(fig_out, bbox_inches="tight")
    plt.close(fig)
    print(f"    กราฟ -> {fig_out.name}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="ทดลองดริฟต์ของอัตราเฟรม")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="เก็บข้อมูลเป็นรอบ ๆ")
    c.add_argument("--outdir", required=True, help="โฟลเดอร์เก็บผล")
    c.add_argument("--interval-s", type=float, default=180.0,
                   help="ช่วงห่างระหว่างรอบ หน่วยวินาที ค่าเริ่มต้น 180")
    c.add_argument("--duration-min", type=float, default=90.0,
                   help="ระยะเวลารวม หน่วยนาที ค่าเริ่มต้น 90")
    c.add_argument("--samples", type=int, default=10_000_000,
                   help="จำนวนตัวอย่างต่อรอบ ค่าเริ่มต้น 10000000 คือ 10 วินาที")
    c.add_argument("--samplerate", default="1m", help="อัตราสุ่ม ค่าเริ่มต้น 1m")
    c.set_defaults(func=cmd_capture)

    a = sub.add_parser("analyse", help="วิเคราะห์ไฟล์ที่เก็บไว้")
    a.add_argument("--outdir", required=True, help="โฟลเดอร์ที่เก็บผล")
    a.add_argument("--channel", type=int, default=2, help="ช่อง INT ค่าเริ่มต้น 2")
    a.add_argument("--samplerate-hz", type=float, default=1e6)
    a.add_argument("--expected", type=float, default=60.0,
                   help="ความถี่ตามสเปกของโหมดที่วัด ค่าเริ่มต้น 60")
    a.add_argument("--duration-note", default="10 s each, 3 min apart")
    a.set_defaults(func=cmd_analyse)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()

# ---------------------------------------------------------------------------
# ข้อจำกัดที่ทราบ
#
# 1. เวลาที่บันทึกคือเวลาที่เริ่มเก็บแต่ละรอบ นับจากตอนที่สคริปต์เริ่มทำงาน
#    ไม่ใช่ตอนที่จ่ายไฟให้บอร์ด ต้องเริ่มสคริปต์ทันทีหลังเสียบ USB
#    และบันทึกเวลาที่เสียบไว้ด้วยมือ เผื่อมีช่วงห่าง
#
# 2. ไม่ได้วัดอุณหภูมิ จึงทดสอบสมมติฐานเรื่องอุณหภูมิได้เพียงทางอ้อม
#    คือดูว่ารูปร่างของเส้นโค้งเข้ากับการเข้าสู่สมดุลความร้อนหรือไม่
#
# 3. ถ้ามีรอบใดล้มเหลว จะข้ามไปโดยบันทึก ok เป็น 0
#    ช่วงเวลาที่เหลือยังคงตรงตามกำหนด เพราะคำนวณจากเวลาเริ่มต้น ไม่ใช่สะสม
#
# 4. สคริปต์ไม่ตรวจว่าเฟิร์มแวร์อยู่โหมดใด ต้องยืนยันด้วย UART ก่อนเริ่ม
#    แล้วปิด PuTTY เพราะ logic analyzer เปิดได้ทีละโปรแกรม
# ---------------------------------------------------------------------------
