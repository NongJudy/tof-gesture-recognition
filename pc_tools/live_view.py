"""live_view.py — หน้าจอดูผลทายท่ามือสดของบอร์ด สำหรับดูเอง/อัดคลิปให้อาจารย์ (โมเดล E5R บนบอร์ด)

หลักสำคัญ: "บอร์ดเป็นคนทาย" — STM32 อ่านเซ็นเซอร์ เตรียมข้อมูล เลื่อนภาพ (re-centring) รันโมเดล แล้วส่งคำตอบมาทาง USB
สคริปต์นี้แค่ "วาดสิ่งที่บอร์ดส่งมา" ไม่โหลดโมเดล ไม่ต้องมี TensorFlow ไม่ได้คิดคำตอบเอง
  - ภาพซ้าย  : สิ่งที่เซ็นเซอร์เห็น (ระยะ 8×8 จากบรรทัด F/S) ใกล้ = สว่าง
  - ตัวใหญ่   : คำตอบนิ่งของบอร์ด (บรรทัด P = โหวต 15 เฟรมล่าสุด ≈ 1 วินาที ในเฟิร์มแวร์)
  - แถบ 7 ท่า : ความน่าจะเป็นของเฟรมล่าสุด (บรรทัด Y ของบอร์ด)
  - แถบล่าง   : ระยะมือ, เฟรมเรต, เวลาที่บอร์ดใช้ต่อเฟรม (บรรทัด Q), ชื่อโมเดลบนบอร์ด (บรรทัดบูต — กด RESET)

(คนละตัวกับ live_demo.py v3 ของการทดสอบ A5 §5.1q ที่ให้ PC รันโมเดลซ้ำด้วย — ตัวนั้นยังอยู่ ไม่ได้แก้)

เฟิร์มแวร์ที่ใช้ได้: MY_TOF_REPLAY = 0, MY_TOF_PRED_LINE = 1 (ค่าปัจจุบัน commit fbca5e4: E5R, RECENTRE 1)

วิธีรัน (ที่ root ของ repo, ปิด PuTTY ก่อน):
    python pc_tools\\live_view.py --port COM3 --flip-lr
  --flip-lr: ทดสอบแล้ว 27 ต.ค. — ไม่ใส่แล้วภาพกลับซ้าย-ขวาเมื่อยืนฝั่งผู้ใช้ (FACTS §5.1y-Y)
  ลองหน้าจอโดยไม่ต่อบอร์ด (เล่นไฟล์บรรทัดที่ live_check บันทึกไว้):
    python pc_tools\\live_view.py --from-log results_live\\<โฟลเดอร์>\\board_lines.txt
  ในหน้าต่าง: กด F = เต็มจอ, Q = ปิด
"""
from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]
NICE = {"Fist": "Fist", "FlatHand": "Flat Hand", "Dislike": "Dislike", "Like": "Like",
        "Love": "Love", "CrossHands": "Cross Hands", "BreakTime": "Break Time"}
COLORS = {"Fist": "#ff6b6b", "FlatHand": "#4dabf7", "Dislike": "#f59f00", "Like": "#51cf66",
          "Love": "#f06595", "CrossHands": "#9775fa", "BreakTime": "#22b8cf"}
VALID_STATUS = (5, 9)               # เหมือนเฟิร์มแวร์ MY_AI_VALID_STATUS_1/2
FW_MIN_MM, FW_MAX_MM = 100, 400     # เฟิร์มแวร์ทายเมื่อมือ (โซนใกล้สุด) อยู่ในช่วงนี้
BEST_LO_MM, BEST_HI_MM = 220, 320   # ช่วงที่ใช้เก็บข้อมูลเทรน (collector 270 ± 50)
VOTE_WINDOW = 15                    # เหมือนเฟิร์มแวร์ MY_AI_VOTE_WINDOW
BUDGET_MS = 33.0
STALE_S = 1.0                       # ไม่มีบรรทัดเกินนี้ = บอร์ดไม่ส่งข้อมูล

BG, PANEL, TEXT, DIM, GOOD, WARN, BAD = "#0f1216", "#181d24", "#e8eaed", "#9aa0a6", "#51cf66", "#fcc419", "#ff6b6b"


@dataclass
class BoardState:
    """สิ่งที่บอร์ดส่งมาล่าสุด (เธรดอ่าน serial เขียน, หน้าจออ่าน — ใช้ lock)"""
    lock: threading.Lock = field(default_factory=threading.Lock)
    dist: np.ndarray | None = None          # 64 ค่า mm ของเฟรมล่าสุดที่มีทั้ง F และ S
    status: np.ndarray | None = None
    frame: int = -1
    y_cls: int | None = None                # คำตอบเฟรมล่าสุด (-1 นอกช่วง, -2 รันพลาด, -3 โมเดลไม่เข้าคู่)
    y_probs: np.ndarray | None = None
    vote_buf: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    p_label: str | None = None              # คำตอบนิ่งจากบรรทัด P ของบอร์ด (None = บอร์ดบอกว่าไม่มีมือ)
    p_seen: bool = False                    # เคยได้บรรทัด P แล้วหรือยัง
    q_ms: deque = field(default_factory=lambda: deque(maxlen=45))
    frame_times: deque = field(default_factory=lambda: deque(maxlen=60))
    model: str | None = None                # จากบรรทัดบูต MY_AI: model=… check=…
    check: str | None = None
    warning: str | None = None
    last_line_t: float = 0.0
    n_lines: int = 0
    _pend: dict = field(default_factory=dict)


def firmware_vote(buf: Iterable[int]) -> int:
    """กติกาเดียวกับ my_tof_send_prediction(): ต้องชนะ 'ไม่มีมือ' แบบมากกว่าจริง เสมอกันให้ท่าเลขน้อยกว่า"""
    b = list(buf)
    if len(b) < VOTE_WINDOW:
        return -9                           # ยังไม่ครบหน้าต่าง (เฟิร์มแวร์ยังไม่ตัดสิน)
    none = sum(c < 0 for c in b)
    best, best_n = -1, none
    for k in range(7):
        n = sum(c == k for c in b)
        if n > best_n:
            best, best_n = k, n
    return best


def parse_line(line: str, st: BoardState, now: float) -> None:
    """อัปเดต state จากบรรทัดเดียวของบอร์ด (ฟังก์ชันล้วน ทดสอบได้โดยไม่ต้องมีบอร์ด)"""
    line = line.strip()
    if not line:
        return
    with st.lock:
        st.last_line_t = now
        st.n_lines += 1
        head = line.split(",", 1)[0]
        try:
            if head in ("F", "S"):
                parts = line.split(",")
                n = int(parts[1])
                vals = np.array([int(v) for v in parts[2:]], dtype=np.int64)
                if vals.size != 64:
                    st.warning = "board sends 4x4 data — the model needs 8x8"
                    return
                slot = st._pend.setdefault(n, {})
                slot[head] = vals
                if "F" in slot and "S" in slot:
                    st.dist, st.status, st.frame = slot["F"], slot["S"], n
                    st.frame_times.append(now)
                    for k in [k for k in st._pend if k <= n]:
                        del st._pend[k]
                for k in [k for k in st._pend if n - k > 30]:
                    del st._pend[k]
            elif head == "Y":
                parts = line.split(",")
                c = int(parts[2])
                st.y_cls = c
                st.vote_buf.append(c if c >= 0 else -1)
                if c >= 0 and len(parts) >= 10:
                    st.y_probs = np.array([float(v) for v in parts[3:10]])
                else:
                    st.y_probs = None
                if c == -3:
                    st.warning = "MODEL CHECK FAILED — switch/model mismatch, board will not predict"
            elif head == "Q":
                parts = line.split(",")
                st.q_ms.append(int(parts[6]) / 1000.0)
            elif head == "P":
                name = line.split(",", 2)[2].strip()
                st.p_label = name if name in CLASSES else None
                st.p_seen = True
            elif line.startswith("MY_AI: model="):
                kv = dict(p.split("=", 1) for p in line[len("MY_AI: "):].split() if "=" in p)
                st.model, st.check = kv.get("model"), kv.get("check")
                if st.check != "OK":
                    st.warning = f"MODEL CHECK = {st.check} — board will not predict"
                elif st.warning and "MODEL" in st.warning:
                    st.warning = None
            elif line.startswith("MY_TOF: REPLAY mode"):
                st.warning = "REPLAY build on the board — flash MY_TOF_REPLAY = 0"
            elif line.startswith("ERROR"):
                st.warning = line[:70]
        except (ValueError, IndexError):
            pass                            # บรรทัดขาด/เพี้ยน (เช่นตอนเพิ่งเปิดพอร์ต) — ข้าม


def snapshot(st: BoardState, now: float) -> dict:
    """คัดค่าที่หน้าจอต้องใช้ออกมาภายใต้ lock แล้วคำนวณค่าที่แสดง"""
    with st.lock:
        d = None if st.dist is None else st.dist.copy()
        s = None if st.status is None else st.status.copy()
        ft = [t for t in st.frame_times if now - t <= 2.0]
        snap = {"dist": d, "status": s, "frame": st.frame, "y_cls": st.y_cls,
                "y_probs": None if st.y_probs is None else st.y_probs.copy(),
                "vote": firmware_vote(st.vote_buf), "votes": list(st.vote_buf), "p_label": st.p_label, "p_seen": st.p_seen,
                "q_last": st.q_ms[-1] if st.q_ms else None, "q_mean": float(np.mean(st.q_ms)) if st.q_ms else None,
                "q_max": float(np.max(st.q_ms)) if st.q_ms else None,
                "hz": (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else None,
                "model": st.model, "check": st.check, "warning": st.warning,
                "stale": (now - st.last_line_t) > STALE_S, "n_lines": st.n_lines}
    hand = None
    if d is not None and s is not None:
        ok = np.isin(s, VALID_STATUS)
        if ok.any():
            hand = int(d[ok].min())
    snap["hand_mm"] = hand
    # คำตอบนิ่ง: ใช้บรรทัด P ของบอร์ดเป็นหลัก (บอร์ดพิมพ์ P เฉพาะตอนคำตอบเปลี่ยน) ; ก่อนเห็น P แรก
    # (เพิ่งเปิดหน้าจอ) ใช้กติกาโหวตเดียวกับเฟิร์มแวร์จากบรรทัด Y ของบอร์ดเอง
    if snap["p_seen"]:
        stable = snap["p_label"]
    else:
        stable = CLASSES[snap["vote"]] if snap["vote"] >= 0 else None
    snap["stable"] = stable
    snap["agree"] = sum(1 for c in snap["votes"] if stable is not None and c == CLASSES.index(stable)) if stable else 0
    return snap


class SerialReader(threading.Thread):
    """อ่านบรรทัดจากบอร์ด (หรือจากไฟล์ log ตามจังหวะ 15 Hz) แล้วส่งเข้า parse_line"""

    def __init__(self, st: BoardState, readline: Callable[[], str], pace_s: float = 0.0) -> None:
        super().__init__(daemon=True)
        self.st, self.readline, self.pace_s = st, readline, pace_s
        self.stop_evt = threading.Event()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                line = self.readline()
            except Exception as exc:  # noqa: BLE001 — พอร์ตหลุด (ถอดสาย) ให้ขึ้นบนจอแทนการพังเงียบ
                with self.st.lock:
                    self.st.warning = f"serial error: {exc}"[:70]
                time.sleep(0.5)
                continue
            if line is None:                # ไฟล์ log หมด
                break
            parse_line(line, self.st, time.monotonic())
            if self.pace_s and line.startswith("F,"):
                time.sleep(self.pace_s)


def log_readline(path: Path, loop: bool) -> Callable[[], str | None]:
    """อ่านไฟล์ board_lines.txt ทีละบรรทัด (วนซ้ำได้) สำหรับลองหน้าจอโดยไม่ต่อบอร์ด"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise SystemExit(f"{path} is empty")
    it = {"i": 0}

    def rd() -> str | None:
        if it["i"] >= len(lines):
            if not loop:
                return None
            it["i"] = 0
        it["i"] += 1
        return lines[it["i"] - 1]
    return rd


def build_figure(flip_lr: bool, flip_ud: bool, source_note: str):
    """สร้างหน้าต่างและคืนฟังก์ชัน draw(snap) ที่อัปเดตทุกอย่าง"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": TEXT,
                         "xtick.color": DIM, "ytick.color": TEXT, "toolbar": "None"})
    fig = plt.figure(figsize=(13.66, 7.68), dpi=100, facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("ToF Hand Posture — live on-board recognition (E5R)")
    except AttributeError:
        pass
    fig.text(0.03, 0.945, "Hand Posture Recognition", fontsize=26, weight="bold", color=TEXT)
    fig.text(0.03, 0.905, "VL53L8CA Time-of-Flight sensor (8×8 zones)  ·  AI model runs on the STM32 NUCLEO-F411RE",
             fontsize=13, color=DIM)
    live_txt = fig.text(0.97, 0.945, "● LIVE", fontsize=16, weight="bold", color=BAD, ha="right")
    fig.text(0.97, 0.908, source_note, fontsize=11, color=DIM, ha="right")

    # ── ซ้าย: สิ่งที่เซ็นเซอร์เห็น ──
    ax_h = fig.add_axes([0.035, 0.17, 0.40, 0.68])
    cmap = plt.get_cmap("inferno_r").copy()
    cmap.set_bad("#2a2f36")
    edges = np.arange(9) - 0.5
    img = ax_h.pcolormesh(edges, edges, np.ma.masked_all((8, 8)), cmap=cmap, norm=Normalize(150, 600),
                          edgecolors=BG, linewidth=1.5)   # pcolormesh วาดเร็วกว่า imshow (ไม่ต้อง resample)
    ax_h.set_xlim(-0.5, 7.5)
    ax_h.set_ylim(7.5, -0.5)                              # แถว 0 อยู่บน
    ax_h.set_aspect("equal")
    ax_h.set_facecolor(BG)
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_color("#3a414b")
    ax_h.set_title("What the sensor sees  (distance per zone, cm — closer = brighter)", fontsize=12, color=DIM, pad=8)
    cell = [[ax_h.text(c, r, "", ha="center", va="center", fontsize=11, color=TEXT) for c in range(8)] for r in range(8)]
    side_txt = fig.text(0.235, 0.135, "", fontsize=11, color=DIM, ha="center")

    # ── ขวาบน: คำตอบนิ่ง ──
    fig.patches.append(plt.Rectangle((0.47, 0.565), 0.50, 0.29, transform=fig.transFigure, color=PANEL, zorder=0))
    fig.text(0.49, 0.815, "BOARD'S ANSWER", fontsize=13, weight="bold", color=DIM)
    ans_txt = fig.text(0.72, 0.69, "", fontsize=60, weight="bold", ha="center", va="center", color=TEXT)
    sub_txt = fig.text(0.72, 0.595, "", fontsize=12, ha="center", color=DIM)

    # ── ขวากลาง: ความน่าจะเป็นของเฟรมนี้ ──
    ax_b = fig.add_axes([0.585, 0.205, 0.33, 0.33], facecolor=BG)
    bars = ax_b.barh(range(7), np.zeros(7), color=[COLORS[c] for c in CLASSES], height=0.66)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(6.6, -0.6)
    ax_b.set_yticks(range(7))
    ax_b.set_yticklabels([NICE[c] for c in CLASSES], fontsize=12)
    ax_b.set_xticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    ax_b.tick_params(length=0)
    pct = [ax_b.text(1.01, k, "", va="center", fontsize=11, color=TEXT, transform=ax_b.get_yaxis_transform())
           for k in range(7)]
    frame_txt = fig.text(0.49, 0.545, "", fontsize=12, color=DIM)

    # ── ล่าง: 4 ช่องสถานะ ──
    tiles = []
    labels = ["Hand distance", "Frame rate", "Board time / frame", "Model on the board"]
    for k, lab in enumerate(labels):
        x0 = 0.47 + k * 0.127
        fig.patches.append(plt.Rectangle((x0, 0.055), 0.118, 0.115, transform=fig.transFigure, color=PANEL, zorder=0))
        fig.text(x0 + 0.008, 0.143, lab, fontsize=9.5, color=DIM)
        v = fig.text(x0 + 0.008, 0.093, "–", fontsize=17, weight="bold", color=TEXT)
        n = fig.text(x0 + 0.008, 0.065, "", fontsize=9, color=DIM)
        tiles.append((v, n))
    fig.text(0.035, 0.014, "All recognition is computed on the STM32F411RE (Cortex-M4, 84 MHz). "
             "The PC only draws what the board sends over USB.", fontsize=10.5, color=DIM)
    warn_txt = fig.text(0.72, 0.868, "", fontsize=11.5, weight="bold", color=BAD, ha="center")

    def draw(sn: dict) -> None:
        live_txt.set_color(BAD if not sn["stale"] and int(time.monotonic() * 2) % 2 == 0 else "#5c2b2b")
        # ภาพเซ็นเซอร์
        if sn["dist"] is not None and sn["status"] is not None:
            d = sn["dist"].reshape(8, 8).astype(float)
            ok = np.isin(sn["status"].reshape(8, 8), VALID_STATUS)
            show = np.where(ok, d, np.nan)
            if flip_lr:
                show, ok, d = show[:, ::-1], ok[:, ::-1], d[:, ::-1]
            if flip_ud:
                show, ok, d = show[::-1], ok[::-1], d[::-1]
            img.set_array(np.ma.masked_invalid(show).ravel())
            for r in range(8):
                for c in range(8):
                    t = cell[r][c]
                    if not ok[r, c]:
                        t.set_text("·")
                        t.set_color(DIM)
                    elif d[r, c] >= 1000:
                        t.set_text("")
                    else:
                        t.set_text(f"{d[r, c] / 10:.0f}")
                        t.set_color("#111111" if d[r, c] < 330 else TEXT)
        side_txt.set_text("top of image = side of the sensor away from the user")
        # คำตอบนิ่ง
        stable = sn["stable"]
        if (sn["warning"] and "MODEL" in sn["warning"]) or sn["y_cls"] == -3:
            ans_txt.set_text("NO PREDICTION")
            ans_txt.set_color(BAD)
            ans_txt.set_fontsize(40)
            sub_txt.set_text("model check failed on the board")
        elif sn["n_lines"] == 0:
            ans_txt.set_text("Waiting …")
            ans_txt.set_color("#5f6670")
            ans_txt.set_fontsize(54)
            sub_txt.set_text("no data from the board yet")
        elif stable is None:
            ans_txt.set_text("No hand")
            ans_txt.set_color("#5f6670")
            ans_txt.set_fontsize(54)
            sub_txt.set_text(f"hold your hand {BEST_LO_MM // 10}–{BEST_HI_MM // 10} cm above the sensor")
        else:
            ans_txt.set_text(NICE[stable])
            ans_txt.set_color(COLORS[stable])
            ans_txt.set_fontsize(60)
            sub_txt.set_text(f"stable answer = majority of the last {VOTE_WINDOW} frames (≈ 1 s)  ·  "
                             f"{sn['agree']}/{len(sn['votes'])} frames agree")
        # แถบความน่าจะเป็น
        p = sn["y_probs"]
        yc = sn["y_cls"]
        for k, b in enumerate(bars):
            v = 0.0 if p is None else float(p[k])
            b.set_width(v)
            b.set_alpha(1.0 if (yc is not None and yc == k) else 0.35)
            pct[k].set_text("" if p is None else f"{v * 100:5.1f} %")
        if yc is None:
            frame_txt.set_text("This frame: waiting for the board …")
        elif yc >= 0:
            frame_txt.set_text(f"This frame (#{sn['frame']}): {NICE[CLASSES[yc]]}  ·  confidence {p[yc] * 100:.1f} %"
                               if p is not None else f"This frame: {NICE[CLASSES[yc]]}")
        elif yc == -1:
            frame_txt.set_text(f"This frame (#{sn['frame']}): no hand in range "
                               f"({FW_MIN_MM // 10}–{FW_MAX_MM // 10} cm) — board does not predict")
        else:
            frame_txt.set_text(f"This frame (#{sn['frame']}): no prediction (code {yc})")
        # ช่องสถานะ
        (v0, n0), (v1, n1), (v2, n2), (v3, n3) = tiles
        h = sn["hand_mm"]
        if h is None or h > FW_MAX_MM:
            v0.set_text("no hand")
            v0.set_color("#5f6670")
            n0.set_text(f"best {BEST_LO_MM // 10}–{BEST_HI_MM // 10} cm")
        else:
            v0.set_text(f"{h / 10:.1f} cm")
            good = BEST_LO_MM <= h <= BEST_HI_MM
            v0.set_color(GOOD if good else WARN)
            n0.set_text("good" if good else ("move hand up" if h < BEST_LO_MM else "move hand closer"))
        v1.set_text("–" if sn["hz"] is None else f"{sn['hz']:.1f} Hz")
        v1.set_color(TEXT)
        n1.set_text("sensor max 15 Hz (8×8)")
        if sn["q_last"] is None:
            v2.set_text("–")
            n2.set_text(f"budget {BUDGET_MS:.0f} ms")
        else:
            v2.set_text(f"{sn['q_mean']:.1f} ms")
            v2.set_color(GOOD if sn["q_max"] <= BUDGET_MS else BAD)
            n2.set_text(f"max {sn['q_max']:.1f} · budget {BUDGET_MS:.0f} ms")
        if sn["model"]:
            v3.set_text(sn["model"].upper())
            v3.set_color(GOOD if sn["check"] == "OK" else BAD)
            n3.set_text(f"model check {sn['check']}")
        else:
            v3.set_text("E5R?")
            v3.set_color(DIM)
            n3.set_text("press RESET to show")
        if sn["stale"]:
            warn_txt.set_text("No data from the board — check USB cable / COM port"
                              if sn["n_lines"] == 0 or not sn["warning"] else sn["warning"])
        else:
            warn_txt.set_text(sn["warning"] or "")

    return fig, draw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--from-log", type=Path, default=None, help="เล่นไฟล์บรรทัดของบอร์ดแทนการต่อบอร์ด (ลองหน้าจอ)")
    ap.add_argument("--no-loop", action="store_true", help="ใช้กับ --from-log: เล่นจบแล้วหยุด")
    ap.add_argument("--flip-lr", action="store_true", help="กลับภาพซ้าย-ขวา")
    ap.add_argument("--flip-ud", action="store_true", help="กลับภาพบน-ล่าง")
    args = ap.parse_args(argv)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    st = BoardState()
    if args.from_log is not None:
        reader = SerialReader(st, log_readline(args.from_log, loop=not args.no_loop), pace_s=1 / 15)
        note = f"replaying {args.from_log.name} (not live)"
    else:
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("pyserial is required: pip install pyserial") from exc
        try:
            ser = serial.Serial(args.port, args.baud, timeout=0.2)
        except serial.SerialException as exc:
            raise SystemExit(f"cannot open {args.port}: {exc}\n  → close PuTTY / other programs using the port, or replug the board")
        ser.reset_input_buffer()
        reader = SerialReader(st, lambda: ser.readline().decode("ascii", errors="replace"))
        note = f"{args.port} @ {args.baud} baud"
    fig, draw = build_figure(args.flip_lr, args.flip_ud, note)
    reader.start()

    def tick(_):
        draw(snapshot(st, time.monotonic()))
        return []

    def on_key(ev):
        if ev.key in ("q", "escape"):
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, tick, interval=66, cache_frame_data=False)  # noqa: F841 — ต้องเก็บอ้างอิงไว้ไม่ให้ถูกเก็บกวาด
    print("Window open — press F for full screen, Q to quit. Press RESET on the board to show the model name.")
    plt.show()
    reader.stop_evt.set()
    if args.from_log is None:
        ser.close()                         # ปล่อยพอร์ตให้โปรแกรมอื่นใช้ต่อ


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. "cannot open COM3" → PuTTY / live_check / collector ยังเปิดพอร์ตอยู่ → ปิด หรือถอดเสียบบอร์ด
#  2. ช่องโมเดลขึ้น "E5R?" → แค่ยังไม่เห็นบรรทัดบูต ; กด RESET (ปุ่มดำ) บนบอร์ดหลังหน้าต่างเปิด จะขึ้น E5R / check OK
#  3. ภาพซ้ายกลับด้าน → ใช้ --flip-lr / --flip-ud (แค่การแสดงผล ไม่มีผลกับการทาย)
#  4. ขึ้น NO PREDICTION / MODEL CHECK → เฟิร์มแวร์กับโมเดลไม่เข้าคู่ (§5.1y-U) ต้องแก้ที่เฟิร์มแวร์ ไม่ใช่ที่สคริปต์
#  5. หน้าจอช้ากว่าบอร์ดเล็กน้อย (วาดทุก 66 ms) — ตัวเลขทุกตัวมาจากบอร์ด ไม่ได้คำนวณใหม่ ยกเว้นระยะมือ (โซนใกล้สุด
#     ที่ status 5/9 = กติกาเดียวกับเฟิร์มแวร์) และเฟรมเรต
# ─────────────────────────────────────────────────────────────────────────
