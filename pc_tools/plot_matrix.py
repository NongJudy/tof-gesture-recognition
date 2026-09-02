"""plot_matrix.py

วิเคราะห์และวาดกราฟผลการวัด latency ทั้ง 8 เงื่อนไข
(8x8 @ 15 Hz และ 4x4 @ 60 Hz อย่างละ 4 ขั้นการปรับปรุง)

ข้อมูลเข้า : timing_8x8_matrix.csv
ผลลัพธ์    : ตารางสรุป + fig_matrix.png / .pdf

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python plot_matrix.py
"""

from __future__ import annotations

import csv
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_FILE = "timing_8x8_matrix.csv"

# เวลา inference ของโมเดล st_cnn2d_handposture บน STM32F401 @ 84 MHz
# ที่มา: ST Model Zoo README  (คนละโมเดลกับ 5.34 ms ของ Bartoli 2024)
INFERENCE_MS = 1.46

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "figure.dpi": 300,
})


def load(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def derive(row: Dict[str, str]) -> Dict[str, float]:
    """คำนวณค่าที่ต้องใช้จากข้อมูลดิบ 1 แถว"""
    i2c = int(row["i2c_total_us"]) / 1000.0        # เวลา I2C รวมต่อเฟรม (ms)
    uart = int(row["uart_us"]) / 1000.0            # เวลา UART ต่อเฟรม (ms)
    work = i2c + uart                              # งานที่ MCU ทำจริงต่อเฟรม
    period = int(row["elapsed_ms"]) / int(row["frames"])   # คาบต่อเฟรม
    return {
        "transfer": int(row["transfer_us"]) / 1000.0,      # เฉพาะก้อนข้อมูลใหญ่
        "i2c": i2c,
        "uart": uart,
        "work": work,
        "period": period,
        "rate": 1000.0 / period,
        "load": work / period * 100.0,             # MCU ใช้กี่ % ของคาบ
        "idle": period - work,                     # เวลาว่างที่เหลือ
    }


def table(rows: List[Dict[str, str]]) -> None:
    print("=" * 100)
    print("TABLE  Per-frame timing across acquisition strategies")
    print("=" * 100)
    hdr = (f"{'':3}{'config':<11}{'fields':<7}{'acq':<9}{'path':<8}"
           f"{'I2C':>6}{'bytes':>7}{'work':>9}{'period':>9}{'rate':>9}{'load':>8}{'idle':>9}")
    print(hdr)
    print("-" * 100)

    for r in rows:
        d = derive(r)
        print(f"{r['run']:3}{r['config']:<11}{r['payload_fields']:<7}"
              f"{r['acquisition']:<9}{r['driver_path']:<8}"
              f"{float(r['i2c_calls']):>6.1f}{int(r['payload_bytes']):>7}"
              f"{d['work']:>8.2f}m{d['period']:>8.2f}m{d['rate']:>8.2f}H"
              f"{d['load']:>7.1f}%{d['idle']:>8.2f}m")
    print("-" * 100)


def deltas(rows: List[Dict[str, str]], group: str) -> None:
    """แยกว่าแต่ละขั้นการปรับปรุงประหยัดเวลาได้เท่าไหร่"""
    sel = [r for r in rows if r["config"] == group]
    if len(sel) < 2:
        return
    print(f"\nการปรับปรุงทีละขั้น — {group}")
    for prev, cur in zip(sel, sel[1:]):
        dp, dc = derive(prev), derive(cur)
        saved = dp["work"] - dc["work"]
        print(f"  {prev['run']} -> {cur['run']}: ประหยัด {saved:6.2f} ms"
              f"   ({dp['work']:.2f} -> {dc['work']:.2f})")
    first, last = derive(sel[0]), derive(sel[-1])
    print(f"  รวม {sel[0]['run']} -> {sel[-1]['run']}: "
          f"ลดลง {(first['work']-last['work'])/first['work']*100:.1f}%"
          f"   เวลาว่างเพิ่ม {last['idle']/first['idle']:.1f} เท่า")


def headroom(rows: List[Dict[str, str]]) -> None:
    print(f"\nพื้นที่สำหรับ inference ({INFERENCE_MS} ms, st_cnn2d บน F401 @ 84 MHz)")
    for r in rows:
        d = derive(r)
        used = (d["work"] + INFERENCE_MS) / d["period"] * 100.0
        fits = int(d["idle"] / INFERENCE_MS)
        print(f"  {r['run']} ({r['config']:<9}): ใช้ {used:5.1f}% ของคาบ"
              f"   รัน inference ได้ {fits:3d} ครั้งในเวลาที่เหลือ")


def plot(rows: List[Dict[str, str]]) -> None:
    groups = ["8x8@15Hz", "4x4@60Hz"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, g in zip(axes, groups):
        sel = [r for r in rows if r["config"] == g]
        labels = [r["run"] for r in sel]
        d = [derive(r) for r in sel]
        x = range(len(sel))

        # ซ้อนแท่ง: ก้อนข้อมูลใหญ่ / I2C ส่วนที่เหลือ / UART / เวลาว่าง
        big = [v["transfer"] for v in d]
        rest = [v["i2c"] - v["transfer"] for v in d]
        ua = [v["uart"] for v in d]
        idle = [v["idle"] for v in d]

        ax.bar(x, big, 0.6, label="I2C payload read",
               color="#2c3e50", edgecolor="black", lw=0.4)
        ax.bar(x, rest, 0.6, bottom=big, label="other I2C",
               color="#7f8c8d", edgecolor="black", lw=0.4)
        ax.bar(x, ua, 0.6, bottom=[a + b for a, b in zip(big, rest)],
               label="UART", color="#bdc3c7", edgecolor="black", lw=0.4)
        ax.bar(x, idle, 0.6,
               bottom=[a + b + c for a, b, c in zip(big, rest, ua)],
               label="idle (available)", color="white",
               edgecolor="black", lw=0.4, hatch="///")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title(g, fontsize=9)
        ax.set_ylabel("per-frame time (ms)" if g == groups[0] else "")
        ax.grid(axis="y", alpha=0.25, ls=":")
        ax.set_axisbelow(True)

        # เขียน % ที่ MCU ใช้ ไว้บนแท่ง
        for xi, v in zip(x, d):
            ax.text(xi, v["period"] * 1.02, f"{v['load']:.0f}%",
                    ha="center", va="bottom", fontsize=7.5)

    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("fig_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig("fig_matrix.pdf", bbox_inches="tight")
    print("\nบันทึกกราฟแล้ว: fig_matrix.png และ fig_matrix.pdf")


def main() -> None:
    rows = load(CSV_FILE)
    table(rows)
    for g in ["8x8@15Hz", "4x4@60Hz"]:
        deltas(rows, g)
    headroom(rows)
    plot(rows)

    print("\nหมายเหตุ")
    print("  - แถว E,F,G วัดด้วยนาฬิกา HSI ซึ่งเดินช้ากว่าความจริง 1.36%")
    print("    ค่าจริงต่ำกว่าที่แสดงประมาณ 1.36% ยังไม่ได้วัดซ้ำด้วย HSE")
    print("  - แถว A-D และ H วัดด้วยนาฬิกาคริสตัล HSE แล้ว")
    print("  - dup = 0 ทุกแถว ตรวจด้วย streamcount ของเซ็นเซอร์เอง")


if __name__ == "__main__":
    main()
