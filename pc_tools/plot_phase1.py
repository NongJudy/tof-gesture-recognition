"""
plot_phase1.py
==============
สร้างกราฟประกอบผลการทดลอง Phase 1

ทำไมป้ายกำกับเป็นภาษาอังกฤษ
---------------------------
matplotlib ที่ติดตั้งตามปกติบน Windows ไม่มีฟอนต์ภาษาไทย
ถ้าใส่ข้อความไทยลงในกราฟ จะแสดงเป็นกล่องสี่เหลี่ยมทั้งหมด
และการติดตั้งฟอนต์เพิ่มทำให้ผู้อื่นรันซ้ำไม่ได้ ซึ่งขัดกับหลักการทำซ้ำได้

การใช้ป้ายภาษาอังกฤษในรูปประกอบ เป็นเรื่องปกติในวิทยานิพนธ์วิศวกรรมไทย
คำอธิบายภาษาไทยเขียนไว้ในคำบรรยายใต้รูปแทน

กราฟที่สร้าง
------------
fig5_drift.png        อัตราเฟรมเปลี่ยนไปตามเวลาหลังเปิดเครื่อง
fig6_period_dist.png  การกระจายของคาบรายเฟรม เทียบสองโหมด
fig7_estimator.png    เทียบตัวประมาณสองแบบ

วิธีรัน
-------
    python plot_phase1.py
    python plot_phase1.py --data-dir ../data/phase1_int --out-dir ../data/phase1_int
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ต้องติดตั้ง numpy ก่อน  พิมพ์:  pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")            # ไม่ต้องเปิดหน้าต่าง เซฟเป็นไฟล์อย่างเดียว
    import matplotlib.pyplot as plt
except ImportError:
    print("ต้องติดตั้ง matplotlib ก่อน  พิมพ์:  pip install matplotlib")
    sys.exit(1)


GUARD_SAMPLES: int = 1000
CAL_PPM: float = -95.408          # ค่าสอบเทียบ logic analyzer เทียบนาฬิกาบอร์ด


# ---------------------------------------------------------------------------
# ข้อมูลดริฟต์ที่วัดได้ 10 ก.ย. 2026
#
# เวลาเป็นนาทีนับจากจุดอ้างอิง 10:58 ซึ่งเป็นการวัดแรกหลังเสียบบอร์ดขยาย
# ค่า ppm เทียบกับความถี่ที่ระบุในดาต้าชีทของแต่ละโหมด
# ทุกค่าคำนวณด้วยวิธีปรับเส้นตรงกับขอบทุกอัน
#
# ถ้าเก็บข้อมูลเพิ่ม ให้เติมแถวที่นี่
# ---------------------------------------------------------------------------
DRIFT_DATA: list[tuple[float, str, float, float]] = [
    # (นาทีจาก 10:58, โหมด, ppm, SD ถ้ามี ไม่มีใส่ nan)
    (0.0,   "4x4", 6038.4, 15.56),
    (47.0,  "4x4", 6945.9, float("nan")),
    (57.0,  "4x4", 6911.1, float("nan")),
    (75.0,  "4x4", 6890.4, 16.94),
    (92.0,  "4x4", 6844.3, 1.68),
    (28.0,  "8x8", 7689.5, 8.22),
    (42.0,  "8x8", 7572.9, float("nan")),
    (78.0,  "8x8", 7582.4, 13.83),
    (82.0,  "8x8", 7567.1, 10.85),
]


def periods_from_file(path: Path, channel: int = 2, samplerate: float = 1e6) -> np.ndarray:
    """อ่านไฟล์ binary แล้วคืนคาบรายเฟรม หน่วยมิลลิวินาที

    ใช้ขอบขาลงของสัญญาณ INT ซึ่งเป็นจังหวะที่เซ็นเซอร์แจ้งว่าเฟรมพร้อม

    Args:
        path: ไฟล์ binary จาก sigrok-cli
        channel: หมายเลขบิตของช่องสัญญาณ
        samplerate: อัตราสุ่มที่ใช้ตอนเก็บ

    Returns:
        อาร์เรย์ของคาบ หน่วยมิลลิวินาที

    Raises:
        FileNotFoundError: หาไฟล์ไม่เจอ
    """
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    raw = np.fromfile(path, dtype=np.uint8)
    core = raw[GUARD_SAMPLES:-GUARD_SAMPLES]
    bits = (core >> channel) & 1
    diff = np.diff(bits.astype(np.int8))
    falling = np.flatnonzero(diff == -1) + 1
    return np.diff(falling) / samplerate * 1000.0


def fig_drift(out: Path) -> None:
    """กราฟที่ 5 อัตราเฟรมเปลี่ยนไปตามเวลา

    ข้อควรระวังในการนำเสนอ
    ----------------------
    ข้อมูลชุดนี้ไม่ได้มาจากการทดลองที่ออกแบบไว้ล่วงหน้า
    แต่เป็นการรวบรวมย้อนหลังจากการวัดที่ทำเพื่อจุดประสงค์อื่น
    จึงมีข้อจำกัดสามข้อที่ต้องแสดงให้ผู้อ่านเห็น

    1. จุดเริ่มต้นที่ศูนย์นาที ไม่ใช่จังหวะที่เปิดเครื่อง
       บอร์ดทำงานมาก่อนหน้านั้นแล้วเป็นเวลาหนึ่ง แต่ไม่ได้บันทึกไว้
       จึงห้ามใช้คำว่า after power-on ในชื่อกราฟ
    2. เวลาเป็นค่าโดยประมาณ อ่านจากเวลาที่ไฟล์ถูกสร้าง
    3. ช่วงระหว่างจุดไม่มีข้อมูล การลากเส้นเชื่อมจะสื่อว่าดริฟต์เป็นเส้นตรง
       ซึ่งเราไม่ได้วัด จึงแสดงเป็นจุดอย่างเดียว ไม่ลากเส้น

    กราฟนี้จึงติดป้ายว่าเป็นผลเบื้องต้น
    และจะถูกแทนที่เมื่อทำการทดลองดริฟต์อย่างเป็นระบบจากบอร์ดเย็นแล้ว
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)

    for mode, colour, marker in (("4x4", "tab:blue", "o"), ("8x8", "tab:red", "s")):
        pts = [(t, p, sd) for t, m, p, sd in DRIFT_DATA if m == mode]
        pts.sort()
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        e = np.array([p[2] for p in pts])

        # แสดงเป็นจุดอย่างเดียว ไม่ลากเส้นเชื่อม
        # เพราะไม่มีข้อมูลระหว่างจุด การลากเส้นจะสื่อสิ่งที่ไม่ได้วัด
        ax.scatter(t, y, marker=marker, color=colour, s=55,
                   label=f"{mode} mode", zorder=3, edgecolor="white", linewidth=0.6)
        ok = ~np.isnan(e)
        if ok.any():
            ax.errorbar(t[ok], y[ok], yerr=e[ok], fmt="none",
                        ecolor=colour, capsize=3, alpha=0.8, zorder=2)

    ax.set_xlabel("Time since first measurement (minutes, approximate)")
    ax.set_ylabel("Frame rate deviation from datasheet (ppm)")
    ax.set_title("Sensor frame rate is not constant over time")
    ax.grid(alpha=0.3)

    # ขยายแกนตั้งเพื่อเว้นที่ให้คำอธิบาย ไม่ให้ทับจุดข้อมูล
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.28, hi + (hi - lo) * 0.05)

    # วางกล่องคำอธิบายในช่องว่างระหว่างกลุ่มข้อมูลสองกลุ่ม
    ax.legend(loc="center", framealpha=0.95)

    # ป้ายบอกว่าเป็นผลเบื้องต้น เพื่อไม่ให้ผู้อ่านตีความเกินข้อมูล
    # วางไว้ใต้กลุ่มข้อมูลทั้งหมด ในพื้นที่ที่เพิ่งขยายเพิ่ม
    ax.text(0.5, 0.02,
            "Preliminary: retrospective data, not a designed experiment. "
            "Times are approximate.\n"
            "No data between points, so no lines are drawn. "
            "Error bars show SD across 3 repeats\n"
            "where available; they are smaller than the markers.",
            transform=ax.transAxes, fontsize=7, color="dimgray",
            va="bottom", ha="center")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  สร้าง {out.name}")


def fig_block_diagram(out: Path) -> None:
    """กราฟที่ 8 แผนผังการวัด

    ทำไมต้องมี
    ----------
    รูปถ่ายแสดงของจริงได้ แต่ไม่แสดงโครงสร้างว่าอะไรต่อกับอะไร
    ผู้อ่านที่ไม่เคยเห็นบอร์ดนี้จะไม่เข้าใจจากรูปถ่ายเพียงอย่างเดียว
    แผนผังจึงจำเป็นในบทวิธีทดลอง

    วาดด้วย matplotlib เพื่อให้สร้างซ้ำได้จากโค้ด
    ไม่ต้องพึ่งโปรแกรมวาดรูปภายนอกที่ผู้อื่นอาจไม่มี
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, sub, colour):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour,
                               edgecolor="black", linewidth=1.1, alpha=0.25))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=7, color="dimgray")

    box(4, 36, 26, 16, "VL53L8CX", "X-NUCLEO-53L8A1 shield", "tab:orange")
    box(40, 36, 26, 16, "STM32F411RE", "NUCLEO-F411RE\n84 MHz from 8 MHz HSE", "tab:blue")
    box(40, 6, 26, 16, "Logic analyser", "FX2LP, 24 MHz crystal", "tab:green")
    box(76, 6, 20, 16, "PC", "sigrok-cli\nPuTTY", "tab:gray")

    def arrow(x1, y1, x2, y2, text, style="-|>", dy=1.6, colour="black"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, linewidth=1.0, color=colour))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center",
                fontsize=7, color=colour)

    # เส้นทางสัญญาณระหว่างเซ็นเซอร์กับ MCU
    arrow(30, 47, 40, 47, "I2C  SCL / SDA  400 kHz", style="<|-|>")
    arrow(30, 41, 40, 41, "INT (data ready)", style="-|>")

    # จุดจิ้มวัด
    ax.plot([53, 53], [36, 22], color="tab:green", linewidth=1.0, linestyle="--")
    ax.text(54, 29, "probe taps on Morpho header\n"
                    "CN10-3 SCL, CN10-5 SDA\n"
                    "CN7-32 INT, CN10-9 GND",
            fontsize=7, va="center", color="tab:green")

    arrow(66, 14, 76, 14, "USB", style="-|>", colour="tab:green")
    arrow(53, 36, 53, 22, "", style="-|>", colour="tab:green")

    # UART ไปยัง PC
    ax.add_patch(FancyArrowPatch((66, 44), (86, 44), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0,
                                 connectionstyle="arc3,rad=0.0", color="tab:blue"))
    ax.add_patch(FancyArrowPatch((86, 44), (86, 22), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="tab:blue"))
    ax.text(76, 46, "UART 460 800 baud", fontsize=7, ha="center", color="tab:blue")

    ax.text(50, 58, "Measurement setup", ha="center", fontsize=11, fontweight="bold")
    ax.text(50, 1.5,
            "Board clock and logic analyser clock are independent crystals; "
            "calibrated against each other to -95.4 ppm.",
            ha="center", fontsize=7, color="dimgray")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  สร้าง {out.name}")


def fig_period_dist(out: Path, f4: Path | None, f8: Path | None) -> None:
    """กราฟที่ 6 การกระจายของคาบรายเฟรม เทียบสองโหมด

    แสดงเป็นสัดส่วนเบี่ยงเบนจากค่ากลาง เพื่อให้เทียบกันได้
    ทั้งที่คาบต่างกันสี่เท่า
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)

    for ax, path, label, colour in (
        (axes[0], f4, "4x4 mode", "tab:blue"),
        (axes[1], f8, "8x8 mode", "tab:red"),
    ):
        if path is None or not path.is_file():
            ax.text(0.5, 0.5, "file not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        per = periods_from_file(path)
        med = float(np.median(per))
        rel = (per - med) / med * 100.0

        ax.hist(rel, bins=40, color=colour, alpha=0.75, edgecolor="none")
        ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Deviation from median period (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nmedian {med:.3f} ms, SD {rel.std(ddof=1):.2f} %")
        ax.grid(alpha=0.3)

    fig.suptitle("Per-frame period distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  สร้าง {out.name}")


def fig_estimator(out: Path) -> None:
    """กราฟที่ 7 เทียบตัวประมาณสองแบบด้วยข้อมูลจริง

    ค่าที่ใช้มาจากผลการวิเคราะห์ไฟล์จริง ไม่ใช่การจำลอง
    """
    labels = ["A1\n(4x4)", "B\n(8x8)", "A2b\n(4x4)",
              "int4x4", "int8x8", "cal 1kHz"]
    sd_span = [44.7, 12.9, 52.9, 36.5, 8.2, 0.35]
    sd_fit = [16.94, 13.83, 1.68, 15.56, 8.22, 0.349]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    ax.bar(x - w / 2, sd_span, w, label="first-to-last edge", color="tab:orange")
    ax.bar(x + w / 2, sd_fit, w, label="least-squares fit", color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Run-to-run SD (ppm)")
    ax.set_title("Least-squares fit reduces measurement scatter")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  สร้าง {out.name}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="สร้างกราฟประกอบผล Phase 1")
    parser.add_argument("--data-dir", default=".", help="โฟลเดอร์ที่เก็บไฟล์ .bin")
    parser.add_argument("--out-dir", default=".", help="โฟลเดอร์ที่จะเซฟรูป")
    args = parser.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("สร้างกราฟ")

    fig_drift(out / "fig5_drift.png")

    f4 = data / "aba_A2b_run1.bin"
    f8 = data / "aba_B_run1.bin"
    try:
        fig_period_dist(out / "fig6_period_dist.png", f4, f8)
    except FileNotFoundError as exc:
        print(f"  ข้ามกราฟที่ 6: {exc}")

    fig_estimator(out / "fig7_estimator.png")
    fig_block_diagram(out / "fig8_setup_diagram.png")

    print("เสร็จ")


if __name__ == "__main__":
    _main()

# ---------------------------------------------------------------------------
# ข้อจำกัดที่ทราบ
#
# 1. ข้อมูลดริฟต์ในตาราง DRIFT_DATA มาจากการวัดที่ไม่ได้ออกแบบเป็นการทดลอง
#    เวลาที่บันทึกเป็นค่าโดยประมาณจากเวลาที่ไฟล์ถูกสร้าง
#    การทดลองดริฟต์อย่างเป็นระบบจากบอร์ดเย็นยังไม่ได้ทำ
#
# 2. กราฟที่ 6 ใช้ไฟล์เดียวต่อโหมด ไม่ได้รวมทั้งสามรอบ
#    เพราะต้องการแสดงการกระจายภายในรอบเดียว ไม่ใช่ระหว่างรอบ
#
# 3. กราฟที่ 7 ใช้ค่าที่คำนวณไว้แล้ว ฝังไว้ในโค้ด
#    ถ้าคำนวณใหม่แล้วค่าเปลี่ยน ต้องแก้ตัวเลขในฟังก์ชัน fig_estimator ด้วย
#
# 4. ป้ายกำกับเป็นภาษาอังกฤษ เพราะ matplotlib มาตรฐานไม่มีฟอนต์ไทย
# ---------------------------------------------------------------------------
