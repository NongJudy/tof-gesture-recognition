"""plot_paper.py

วาดกราฟเปรียบเทียบแบบมาตรฐานงานตีพิมพ์ (IEEE two-column) จาก log ทั้ง 3 เงื่อนไข
ต่างจาก plot_comparison.py ตรงที่:
    - ฟอนต์ serif ขนาดเล็ก ตามธรรมเนียมเปเปอร์
    - มี error bar (+/- 1 SD) ทุกแท่ง
    - ใช้ลายเส้น (hatch) ควบคู่สี พิมพ์ขาวดำแล้วยังแยกออก
    - กว้าง 7.16 นิ้ว = ความกว้างเต็มหน้าของ IEEE two-column
    - พิมพ์ตารางตัวเลขออกมาให้ก๊อปไปใส่รายงานได้เลย

ผลลัพธ์:
    fig_latency.png   (300 dpi สำหรับดู/ส่งไลน์)
    fig_latency.pdf   (เวกเตอร์ สำหรับใส่เปเปอร์ ไม่แตกเมื่อซูม)

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python plot_paper.py
"""

from __future__ import annotations

import os
import statistics as st
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARMUP_FRAMES = 2          # ตัดเฟรม setup ของทุกรอบการรัน
RANGING_HZ = 15.0          # ความถี่วัดของเซ็นเซอร์ ใช้คำนวณงบเวลา
INFERENCE_MS = 5.34        # inference CNN บน Cortex-M4 84 MHz (Bartoli 2024, Table V)

CONDITIONS: List[Tuple[str, str]] = [
    ("timing_before_blocking.log", "Blocking\n(polling)"),
    ("timing_after_uart_irq.log", "UART\ninterrupt"),
    ("timing_after_int.log", "UART IRQ +\nINT pin (EXTI)"),
]

# ─── รูปแบบตามธรรมเนียมเปเปอร์ ────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,
    "figure.dpi": 300,
})

# สีอ่อน + ลายเส้น เพื่อให้พิมพ์ขาวดำแล้วยังแยกกันออก
STYLE = [
    ("#4a6d8c", "//"),    # I2C data transfer
    ("#c8a45c", "\\\\"),  # I2C status polling
    ("#8aa87b", "xx"),    # UART transmit
]


def load(path: str) -> Dict[str, float]:
    """อ่าน log 1 ไฟล์ คืนค่าเฉลี่ยและ SD ของแต่ละส่วน (หน่วย ms)"""
    rows: List[Tuple[int, int, int, int]] = []
    prev, run_len = -1, 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("T,"):
                continue
            p = line.strip().split(",")
            if len(p) != 8:
                continue
            try:
                frame, calls, _b, rd_us, _mb, max_us, uart_us = (int(v) for v in p[1:])
            except ValueError:
                continue
            if frame <= prev:      # เลขเฟรมย้อนกลับ = เริ่มรอบใหม่
                run_len = 0
            prev = frame
            run_len += 1
            if run_len > WARMUP_FRAMES:
                rows.append((calls, rd_us, max_us, uart_us))

    if len(rows) < 2:
        raise ValueError(f"ข้อมูลน้อยเกินไปในไฟล์ {path}")

    tr = [r[2] / 1000 for r in rows]                    # data transfer
    po = [(r[1] - r[2]) / 1000 for r in rows]           # status polling
    ua = [r[3] / 1000 for r in rows]                    # UART
    to = [(r[1] + r[3]) / 1000 for r in rows]           # total
    ca = [float(r[0]) for r in rows]

    return {
        "transfer": st.mean(tr), "transfer_sd": st.stdev(tr),
        "poll": st.mean(po), "poll_sd": st.stdev(po),
        "uart": st.mean(ua), "uart_sd": st.stdev(ua),
        "total": st.mean(to), "total_sd": st.stdev(to),
        "calls": st.mean(ca), "calls_sd": st.stdev(ca),
        "n": float(len(rows)),
    }


def print_table(labels: List[str], data: List[Dict[str, float]], budget: float) -> None:
    """พิมพ์ตารางตัวเลขแบบก๊อปไปวางในรายงานได้เลย"""
    flat = [l.replace("\n", " ") for l in labels]
    w = max(len(f) for f in flat) + 2

    print("\n" + "=" * (w + 58))
    print("TABLE  Per-frame latency breakdown (mean +/- SD)")
    print("=" * (w + 58))
    print(f"{'Condition':<{w}}{'Transfer':>13}{'Polling':>13}{'UART':>12}{'Total':>13}")
    print("-" * (w + 58))
    for f, d in zip(flat, data):
        print(f"{f:<{w}}"
              f"{d['transfer']:>7.2f}+/-{d['transfer_sd']:<4.2f}"
              f"{d['poll']:>7.2f}+/-{d['poll_sd']:<4.2f}"
              f"{d['uart']:>6.2f}+/-{d['uart_sd']:<4.2f}"
              f"{d['total']:>7.2f}+/-{d['total_sd']:<4.2f}")
    print("-" * (w + 58))

    print(f"\n{'Condition':<{w}}{'I2C calls':>14}{'Frames':>9}"
          f"{'Budget':>9}{'+CNN':>8}")
    print("-" * (w + 40))
    for f, d in zip(flat, data):
        t = d["total"]
        print(f"{f:<{w}}{d['calls']:>7.1f}+/-{d['calls_sd']:<5.1f}"
              f"{int(d['n']):>9}{t/budget*100:>8.1f}%"
              f"{(t+INFERENCE_MS)/budget*100:>7.1f}%")
    print("-" * (w + 40))


def main() -> None:
    data, labels = [], []
    for fname, label in CONDITIONS:
        if not os.path.exists(fname):
            raise SystemExit(f"ไม่พบไฟล์ {fname} — ต้องมีครบทั้ง 3 ไฟล์")
        data.append(load(fname))
        labels.append(label)

    budget = 1000.0 / RANGING_HZ
    x = list(range(len(data)))
    n_total = int(sum(d["n"] for d in data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ── (a) เวลาต่อเฟรม แยกเป็น 3 ส่วน ────────────────────────
    tr = [d["transfer"] for d in data]
    po = [d["poll"] for d in data]
    ua = [d["uart"] for d in data]
    base = [a + b for a, b in zip(tr, po)]
    segs = [("I2C data transfer", tr, 0), ("I2C status polling", po, 0),
            ("UART transmit", ua, 0)]

    for i, (name, vals, _) in enumerate(segs):
        bottom = [0] * 3 if i == 0 else (tr if i == 1 else base)
        color, hatch = STYLE[i]
        ax1.bar(x, vals, 0.5, bottom=bottom, label=name,
                color=color, hatch=hatch, edgecolor="black", linewidth=0.6)

    # error bar บนยอดแท่ง = ความผันผวนของเวลารวม
    ax1.errorbar(x, [d["total"] for d in data],
                 yerr=[d["total_sd"] for d in data],
                 fmt="none", ecolor="black", capsize=3, elinewidth=0.8)

    ax1.axhline(budget, ls="--", lw=0.9, color="black")
    ax1.text(0.55, budget, f"frame budget at {RANGING_HZ:.0f} Hz "
             f"= {budget:.1f} ms", ha="center", va="center", fontsize=7.5,
             style="italic",
             bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))

    for i, d in enumerate(data):
        ax1.text(i, d["total"] + d["total_sd"] + 2.0,
                 f"{d['total']:.2f}", ha="center", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Time per frame (ms)")
    ax1.set_ylim(0, budget * 1.42)
    ax1.set_xlabel("(a)", labelpad=6)
    ax1.legend(loc="upper right", framealpha=1.0, edgecolor="black")
    ax1.grid(axis="y", alpha=0.25, ls=":")
    ax1.set_axisbelow(True)

    # ── (b) จำนวน transaction ต่อเฟรม ─────────────────────────
    calls = [d["calls"] for d in data]
    ax2.bar(x, calls, 0.5, yerr=[d["calls_sd"] for d in data],
            color="#4a6d8c", hatch="//", edgecolor="black", linewidth=0.6,
            error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 0.8})

    for i, v in enumerate(calls):
        ax2.text(i, v + max(calls) * 0.045, f"{v:.1f}",
                 ha="center", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("I2C transactions per frame")
    ax2.set_ylim(0, max(calls) * 1.22)
    ax2.set_xlabel("(b)", labelpad=6)
    ax2.grid(axis="y", alpha=0.25, ls=":")
    ax2.set_axisbelow(True)

    # คำบรรยายใต้รูป ตามธรรมเนียมเปเปอร์
    caption = (
        f"Fig. 1.  Per-frame latency of the VL53L8CX-to-STM32F411RE link under three "
        f"driver configurations.\n(a) Latency breakdown; (b) number of I$^2$C "
        f"transactions per frame. Error bars show $\\pm$1 SD.\n"
        f"8$\\times$8 resolution at {RANGING_HZ:.0f} Hz, I$^2$C 400 kHz, "
        f"CPU 84 MHz, n = {n_total} frames."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.8, wrap=True)

    fig.tight_layout()
    fig.savefig("fig_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig("fig_latency.pdf", bbox_inches="tight")
    print("บันทึกแล้ว: fig_latency.png (ส่งไลน์)  และ  fig_latency.pdf (ใส่เปเปอร์)")

    print_table(labels, data, budget)


if __name__ == "__main__":
    main()
