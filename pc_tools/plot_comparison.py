"""plot_comparison.py

อ่านไฟล์ log ทั้ง 3 เงื่อนไข แล้ววาดกราฟเปรียบเทียบในรูปเดียว
ใช้ส่งอาจารย์ และใช้เป็นรูปในรายงาน/เปเปอร์ได้เลย

ผลลัพธ์: comparison.png  (2 แผง)
    ซ้าย  = เวลาต่อเฟรม แยกเป็น 3 ส่วน + เส้นงบเวลาที่ 15 Hz
    ขวา   = จำนวนครั้งที่เรียก I2C ต่อเฟรม

วิธีรัน (ในโฟลเดอร์ pc_tools):
    python plot_comparison.py
"""

from __future__ import annotations

import os
import statistics as st
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")               # เซฟไฟล์อย่างเดียว ไม่เปิดหน้าต่าง
import matplotlib.pyplot as plt

# ตัดเฟรม warm-up ของทุกรอบการรัน (เฟรมแรกมีงาน setup ปน)
WARMUP_FRAMES = 2

# ความถี่วัดของเซ็นเซอร์ ใช้คำนวณงบเวลาต่อเฟรม
RANGING_HZ = 15.0

# เวลา inference ของ CNN บน Cortex-M4 84 MHz (Bartoli et al. 2024, Table V)
INFERENCE_MS = 5.34

# ไฟล์ log ของแต่ละเงื่อนไข: (ชื่อไฟล์, ป้ายบนกราฟ)
CONDITIONS: List[Tuple[str, str]] = [
    ("timing_before_blocking.log", "Blocking\n(polling)"),
    ("timing_after_uart_irq.log", "+ UART\ninterrupt"),
    ("timing_after_int.log", "+ INT pin\n(EXTI)"),
]


def load(path: str) -> Dict[str, float]:
    """อ่าน log 1 ไฟล์ คืนค่าเฉลี่ยของเวลาแต่ละส่วน (ms) และจำนวนครั้งที่เรียก I2C"""
    rows: List[Tuple[int, int, int, int, int]] = []
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

            # เลขเฟรมย้อนกลับ = เริ่มรอบใหม่ ต้องตัด warm-up อีกครั้ง
            if frame <= prev:
                run_len = 0
            prev = frame
            run_len += 1
            if run_len > WARMUP_FRAMES:
                rows.append((frame, calls, rd_us, max_us, uart_us))

    if not rows:
        raise ValueError(f"ไม่พบข้อมูลที่ใช้ได้ในไฟล์ {path}")

    transfer = st.mean(r[3] for r in rows) / 1000.0          # โอนข้อมูลก้อนใหญ่
    poll = st.mean(r[2] - r[3] for r in rows) / 1000.0       # เวลาที่เหลือ = poll
    uart = st.mean(r[4] for r in rows) / 1000.0
    return {
        "transfer": transfer,
        "poll": poll,
        "uart": uart,
        "total": transfer + poll + uart,
        "calls": st.mean(r[1] for r in rows),
        "n": float(len(rows)),
    }


def main() -> None:
    data, labels = [], []
    for fname, label in CONDITIONS:
        if not os.path.exists(fname):
            raise SystemExit(f"ไม่พบไฟล์ {fname} — ต้องมีครบทั้ง 3 ไฟล์")
        data.append(load(fname))
        labels.append(label)
        d = data[-1]
        print(f"{fname:32} n={int(d['n']):5d}  total={d['total']:6.2f} ms  "
              f"calls={d['calls']:6.1f}")

    budget = 1000.0 / RANGING_HZ
    x = range(len(data))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # ── แผงซ้าย: เวลาต่อเฟรม แยกส่วน ────────────────────────
    transfer = [d["transfer"] for d in data]
    poll = [d["poll"] for d in data]
    uart = [d["uart"] for d in data]
    base = [t + p for t, p in zip(transfer, poll)]

    ax1.bar(x, transfer, 0.55, label="I2C data transfer", color="#2c6fbb")
    ax1.bar(x, poll, 0.55, bottom=transfer, label="I2C status polling", color="#e2a33c")
    ax1.bar(x, uart, 0.55, bottom=base, label="UART transmit", color="#8fbf6b")

    ax1.axhline(budget, ls="--", lw=1.2, color="#c0392b")
    ax1.text(-0.42, budget + 1.0,
             f"frame budget @ {RANGING_HZ:.0f} Hz = {budget:.1f} ms",
             ha="left", color="#c0392b", fontsize=9)

    for i, d in enumerate(data):
        ax1.text(i, d["total"] + 1.2, f"{d['total']:.2f} ms",
                 ha="center", fontweight="bold", fontsize=10)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Time per frame (ms)")
    ax1.set_ylim(0, budget * 1.30)
    ax1.set_title("(a) Per-frame latency breakdown")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # ── แผงขวา: จำนวนครั้งที่เรียก I2C ต่อเฟรม ──────────────
    calls = [d["calls"] for d in data]
    bars = ax2.bar(x, calls, 0.55, color=["#2c6fbb", "#2c6fbb", "#c0392b"])
    for b, v in zip(bars, calls):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(calls) * 0.02,
                 f"{v:.0f}", ha="center", fontweight="bold", fontsize=10)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("I2C transactions per frame")
    ax2.set_ylim(0, max(calls) * 1.18)
    ax2.set_title("(b) I2C transactions per frame")
    ax2.grid(axis="y", alpha=0.3)

    n_total = int(sum(d["n"] for d in data))
    fig.suptitle(
        f"VL53L8CX on NUCLEO-F411RE — 8x8 @ {RANGING_HZ:.0f} Hz, "
        f"I2C 400 kHz, CPU 84 MHz  (n = {n_total} frames)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("comparison.png", dpi=200)
    print("\nบันทึกกราฟแล้ว: comparison.png")

    # สรุปสัดส่วนงบเวลา เผื่อใช้อ้างในรายงาน
    print("\nสัดส่วนงบเวลาต่อเฟรม")
    for lab, d in zip(labels, data):
        t = d["total"]
        print(f"  {lab.replace(chr(10), ' '):22} {t:6.2f} ms = {t/budget*100:5.1f}%"
              f"   + inference {INFERENCE_MS} ms -> {(t+INFERENCE_MS)/budget*100:5.1f}%")


if __name__ == "__main__":
    main()
