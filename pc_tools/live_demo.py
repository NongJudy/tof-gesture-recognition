"""live_demo.py — ทดสอบท่ามือสดๆ หน้าจอใหญ่ อ่านง่าย (ลองเองหรืออัดคลิป)  v3

v3 (26 ต.ค. 2026): เลือกได้ว่าใครเป็นคนทาย  --source
  pc    (ค่าเริ่มต้น เหมือน v2) — PC ทายด้วยไฟล์ --model
  board — ใช้ผลทายของ "บอร์ด" เอง (บรรทัด Y, เฟิร์มแวร์ MY_TOF_PRED_LINE = 1) PC แค่แสดงผล ไม่ต้องโหลด TensorFlow
  both  — บอร์ดทาย (แสดงผลหลัก) + PC ทายคู่กันทุกเฟรม แล้วแสดงว่า "บอร์ดตรงกับ PC กี่เฟรม"

ลำดับการทำงาน:
  1. ยังไม่มีมือ / มืออยู่นอกช่วง 220–320 มม.  → แสดง "PLACE YOUR HAND" หรือ "MOVE CLOSER / FARTHER" ไม่ทาย
  2. มือเข้าช่วงแล้ว → ต้องถือนิ่งอยู่ในช่วงต่อเนื่อง 1 วินาที (มีแถบนับ) ถ้าหลุดช่วงนับใหม่
  3. ครบ 1 วินาที → เริ่มทาย ผลมาจากการโหวตเฟรมหลังจากนั้น (สูงสุด 15 เฟรมล่าสุด)
  4. มือออกนอกช่วงติดกัน 8 เฟรม (~0.5 วิ) → กลับไปข้อ 1 (เอามือออกแล้วเปลี่ยนท่าได้เลย)

preprocessing เหมือนตอนฝึกทุกขั้น (st_dataset.py): status 5/9, ตัดฉากหลัง 120 มม., normalize 295/196, 281/452

วิธีรัน (ที่ root ของ repo, ปิด PuTTY ก่อน):
    python pc_tools\\live_demo.py --port COM3 --model results_phase55_final\\deploy\\E5.keras
    python pc_tools\\live_demo.py --port COM3 --source board              (บอร์ดทายเอง)
    python pc_tools\\live_demo.py --port COM3 --source both --model results_phase55_final\\deploy\\E5.keras
ปิดโปรแกรม: กด q หรือปิดหน้าต่าง
ตัวเลือก: --window-lo 220 --window-hi 320 --hold 1.0 --vote-window 15

ข้อควรรู้: ผลที่เห็นเป็นการลองดู ไม่ใช่ผลการทดลอง (มือผู้จัดทำอยู่ในข้อมูลที่ใช้ฝึก E5)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]  # ลำดับของ train_st_cnn2d.py
VALID_STATUS = (5, 9)
MIN_D, MAX_D, BG_D = 100.0, 400.0, 120.0
DIST_MEAN, DIST_STD = 295.0, 196.0
SIG_MEAN, SIG_STD = 281.0, 452.0


def parse_line(line: str):
    """'F,12,...' / 'S,12,...' / 'G,12,...' / 'Y,12,<ท่า>,<p0..p6>' → (ชนิด, เลขเฟรม, ค่า) หรือ None"""
    parts = line.strip().split(",")
    if len(parts) < 2 or parts[0] not in ("F", "S", "G", "Y"):
        return None
    try:
        frame_no = int(parts[1])
        values = [float(v) for v in parts[2:]]
    except ValueError:
        return None
    return parts[0], frame_no, values


def preprocess(dist: np.ndarray, status: np.ndarray, sig: np.ndarray):
    """คืน (x[8,8,2], ระยะมือ มม.) หรือ (None, ระยะมือ/None) ถ้าไม่มีมือในช่วง 100–400 มม."""
    valid = np.isin(status, VALID_STATUS)
    if not valid.any():
        return None, None
    hand = float(dist[valid].min())
    if not (MIN_D <= hand <= MAX_D):
        return None, hand
    keep = valid & (dist <= hand + BG_D)
    d = np.where(keep, dist, 4000.0)
    s = np.where(keep, sig, 0.0)
    x = np.stack([((d - DIST_MEAN) / DIST_STD).reshape(8, 8), ((s - SIG_MEAN) / SIG_STD).reshape(8, 8)], axis=-1)
    return x.astype(np.float32), hand


class FrameReader:
    """อ่าน serial แบบไม่ค้าง: ดึงทุกบรรทัดที่มี ประกอบ F/S/G ของเฟรมเดียวกัน คืนเฟรมที่ครบ"""

    def __init__(self, ser, need_y: bool = False):
        self.ser = ser
        self.kinds = "FSGY" if need_y else "FSG"     # v3: โหมดบอร์ดต้องได้บรรทัด Y ของเฟรมเดียวกันด้วย
        self.buf = b""
        self.pending: dict[int, dict[str, list[float]]] = {}
        self.wrong_mode = False

    def poll(self) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        n = self.ser.in_waiting
        if n:
            self.buf += self.ser.read(n)
        out = []
        *lines, self.buf = self.buf.split(b"\n")
        for raw in lines:
            p = parse_line(raw.decode("utf-8", errors="ignore"))
            if p is None:
                continue
            kind, fno, vals = p
            if kind == "Y":
                if "Y" not in self.kinds:
                    continue
            else:
                if len(vals) == 16:
                    self.wrong_mode = True
                if len(vals) != 64:
                    continue
            slot = self.pending.setdefault(fno, {})
            slot[kind] = vals
            if all(k in slot for k in self.kinds):
                out.append((fno, np.array(slot["F"]), np.array(slot["S"]), np.array(slot["G"]), slot.get("Y")))
                del self.pending[fno]
            for k in [k for k in self.pending if fno - k > 30]:
                del self.pending[k]
        return out


BG, PANEL, FG, MUTED = "#101418", "#1A2027", "#F2F4F7", "#8A96A3"
OK, WAIT, BAD = "#4FC3F7", "#FFB74D", "#EF5350"


def main(argv=None, serial_factory=None, predictor=None, max_seconds=None, save_png=None, script=None) -> None:
    ap = argparse.ArgumentParser(description="Live hand-posture demo (big display)")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--model", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--source", choices=("pc", "board", "both"), default="pc",
                    help="ใครทาย: pc = PC ทาย, board = บอร์ดทาย (บรรทัด Y), both = บอร์ดทาย + PC ทายเทียบ")
    ap.add_argument("--vote-window", type=int, default=15)
    ap.add_argument("--window-lo", type=float, default=220.0, help="ระยะมือใกล้สุดที่เริ่มทาย (มม.)")
    ap.add_argument("--window-hi", type=float, default=320.0, help="ระยะมือไกลสุดที่เริ่มทาย (มม.)")
    ap.add_argument("--hold", type=float, default=1.0, help="ต้องถือมือในช่วงนิ่งกี่วินาทีก่อนเริ่มทาย")
    ap.add_argument("--leave-frames", type=int, default=8, help="หลุดช่วงติดกันกี่เฟรมถึงนับว่าเอามือออก")
    args = ap.parse_args(argv)

    import matplotlib
    if save_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if predictor is None and args.source != "board":
        import tensorflow as tf
        if not args.model.exists():
            raise SystemExit(f"ไม่พบไฟล์โมเดล: {args.model}")
        print(f"Loading model {args.model} ...")
        model = tf.keras.models.load_model(str(args.model), compile=False)
        if model.output_shape[-1] != len(CLASSES):
            raise SystemExit(f"model has {model.output_shape[-1]} outputs, expected 7")
        fn = tf.function(lambda x: model(x, training=False))
        predictor = lambda x: fn(tf.constant(x[None, ...])).numpy()[0]   # noqa: E731
        predictor(np.zeros((8, 8, 2), np.float32))          # อุ่นเครื่องครั้งแรก ไม่ให้ภาพกระตุกตอนเริ่ม

    if serial_factory is None:
        import serial
        serial_factory = lambda: serial.Serial(args.port, args.baud, timeout=0)   # noqa: E731
    ser = serial_factory()
    reader = FrameReader(ser, need_y=args.source != "pc")

    # ---------------- หน้าจอ ----------------
    plt.rcParams.update({"toolbar": "None", "font.family": "DejaVu Sans", "text.color": FG,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": FG})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("ToF Hand Posture Recognition — live")
    except Exception:  # noqa: BLE001
        pass
    fig.text(0.03, 0.95, "ToF Hand Posture Recognition", fontsize=16, weight="bold", color=FG, va="center")
    who = {"pc": "inference on the PC (" + args.model.name + ")",
           "board": "inference ON THE BOARD (E5, STM32F411RE)",
           "both": "inference ON THE BOARD (E5)  ·  PC re-checks every frame"}[args.source]
    fig.text(0.03, 0.91, "VL53L8CA 8×8  ·  NUCLEO-F411RE  ·  " + who, fontsize=11, color=MUTED, va="center")

    # กล่องผลลัพธ์หลัก
    ax_main = fig.add_axes([0.03, 0.40, 0.60, 0.46]); ax_main.axis("off")
    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                     fc=PANEL, ec="none", transform=ax_main.transAxes))
    status = ax_main.text(0.5, 0.86, "", ha="center", va="center", fontsize=15, color=MUTED, weight="bold")
    big = ax_main.text(0.5, 0.52, "", ha="center", va="center", fontsize=56, weight="bold", color=FG)
    sub = ax_main.text(0.5, 0.20, "", ha="center", va="center", fontsize=16, color=MUTED)
    bar_bg = Rectangle((0.15, 0.07), 0.70, 0.035, fc="#2A323C", ec="none")
    bar_fg = Rectangle((0.15, 0.07), 0.0, 0.035, fc=OK, ec="none")
    ax_main.add_patch(bar_bg); ax_main.add_patch(bar_fg)

    # มาตรวัดระยะมือ
    ax_g = fig.add_axes([0.03, 0.27, 0.60, 0.07]); ax_g.set_facecolor(BG)
    ax_g.set_xlim(100, 450); ax_g.set_ylim(0, 1); ax_g.set_yticks([])
    for sp in ax_g.spines.values():
        sp.set_visible(False)
    ax_g.add_patch(Rectangle((100, 0.35), 350, 0.3, fc="#2A323C", ec="none"))
    ax_g.add_patch(Rectangle((args.window_lo, 0.35), args.window_hi - args.window_lo, 0.3, fc="#1E4E3A", ec="none"))
    marker, = ax_g.plot([], [], marker="v", ms=16, color=FG, ls="")
    ax_g.set_xticks([100, args.window_lo, args.window_hi, 400])
    ax_g.tick_params(axis="x", colors=MUTED, length=0, labelsize=11)
    ax_g.set_xlabel("hand distance (mm)  ·  green = recognition range", fontsize=11, color=MUTED)

    # ความน่าจะเป็นทั้ง 7 ท่า
    ax_bar = fig.add_axes([0.13, 0.04, 0.50, 0.17]); ax_bar.set_facecolor(BG)
    bars = ax_bar.barh(range(7), [0] * 7, color="#2A323C", height=0.7)
    ax_bar.set_yticks(range(7), CLASSES, fontsize=10)
    ax_bar.invert_yaxis(); ax_bar.set_xlim(0, 100); ax_bar.set_xticks([])
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    pct_txt = [ax_bar.text(101, i, "", va="center", fontsize=9, color=MUTED) for i in range(7)]

    # ภาพระยะ 8×8
    ax_img = fig.add_axes([0.67, 0.20, 0.30, 0.60]); ax_img.set_facecolor(BG)
    im = ax_img.imshow(np.full((8, 8), 4000.0), cmap="magma_r", vmin=150, vmax=450)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_color("#2A323C")
    ax_img.set_title("sensor view (distance)", color=MUTED, fontsize=12)
    fig.text(0.82, 0.14, "bright = near, dark = far / background", ha="center", fontsize=10, color=MUTED)
    fig.text(0.82, 0.06, "q = quit", ha="center", fontsize=10, color=MUTED)

    quit_flag = {"q": False}
    fig.canvas.mpl_connect("key_press_event", lambda ev: quit_flag.update(q=ev.key in ("q", "escape")))
    fig.canvas.mpl_connect("close_event", lambda ev: quit_flag.update(q=True))

    hold_frames = max(1, round(args.hold * 15))          # 8×8 ≈ 15 เฟรม/วินาที
    votes: deque[int] = deque(maxlen=args.vote_window)
    st = {"phase": "idle", "in_cnt": 0, "out_cnt": 0, "hand": None, "probs": np.zeros(7), "cmp": 0, "agree": 0}
    last_draw, t0, last_frame_t = 0.0, time.time(), time.time()
    plt.show(block=False)
    print("Ready. Put your hand 22–32 cm above the sensor and hold still.  q = quit")

    def step(dist, status_arr, sig, yv=None):
        x, hand = preprocess(dist, status_arr, sig)
        st["hand"] = hand
        disp = np.where(np.isin(status_arr, VALID_STATUS), dist, 4000.0).reshape(8, 8)
        im.set_data(disp)
        inside = x is not None and args.window_lo <= hand <= args.window_hi
        if not inside:
            st["out_cnt"] += 1
            st["in_cnt"] = 0
            if st["phase"] == "armed" and st["out_cnt"] < args.leave_frames:
                return                                    # หลุดช่วงสั้นๆ ระหว่างทาย: ข้ามเฟรมนี้ ไม่รีเซ็ต
            st["phase"] = "idle"
            votes.clear(); st["probs"] = np.zeros(7)
            return
        st["out_cnt"] = 0
        if st["phase"] != "armed":
            st["in_cnt"] += 1
            st["phase"] = "holding"
            if st["in_cnt"] >= hold_frames:
                st["phase"] = "armed"
                votes.clear()
            return
        if args.source == "pc":
            st["probs"] = np.asarray(predictor(x), dtype=float)
        else:
            # v3: ผลของบอร์ด — Y,<เฟรม>,<ท่า>,<p0..p6> ; -1 = บอร์ดไม่ทายเฟรมนี้ (ไม่ควรเกิดเมื่อมืออยู่ในช่วง)
            if yv is None or len(yv) < 8 or yv[0] < 0:
                return
            st["probs"] = np.asarray(yv[1:8], dtype=float)
            if args.source == "both":
                st["cmp"] += 1
                st["agree"] += int(int(np.argmax(predictor(x))) == int(yv[0]))
        votes.append(int(st["probs"].argmax()))

    while not quit_flag["q"]:
        if script is not None:                            # สำหรับทดสอบ: ป้อนเฟรมจำลอง
            frames = script(time.time() - t0)
        else:
            frames = reader.poll()
        for fr in frames:
            last_frame_t = time.time()
            step(fr[1], fr[2], fr[3], fr[4] if len(fr) > 4 else None)

        now = time.time()
        if now - last_draw > 0.066:
            hand = st["hand"]
            bar_fg.set_width(0); bar_fg.set_color(OK); bar_fg.set_visible(False)
            if reader.wrong_mode:
                status.set_text("FIRMWARE"); big.set_text("4×4 MODE"); big.set_color(BAD)
                sub.set_text("board sends 4×4 — model needs 8×8")
            elif now - last_frame_t > 2.0:
                status.set_text("NO SIGNAL"); big.set_text("NO DATA"); big.set_color(BAD)
                sub.set_text("no frames — check COM port, close PuTTY"
                             + ("; board mode needs firmware MY_TOF_PRED_LINE = 1" if args.source != "pc" else ""))
            elif st["phase"] == "idle":
                status.set_text("WAITING")
                if hand is None or hand > 400:
                    big.set_text("PLACE HAND"); sub.set_text(f"hold your hand {args.window_lo/10:.0f}–{args.window_hi/10:.0f} cm above the sensor")
                elif hand > args.window_hi:
                    big.set_text("MOVE CLOSER"); sub.set_text(f"hand at {hand:.0f} mm")
                else:
                    big.set_text("MOVE FARTHER"); sub.set_text(f"hand at {hand:.0f} mm")
                big.set_color(MUTED)
            elif st["phase"] == "holding":
                frac = min(1.0, st["in_cnt"] / hold_frames)
                status.set_text("HAND DETECTED"); big.set_text("HOLD STILL"); big.set_color(WAIT)
                sub.set_text(f"starting in {max(0.0, args.hold * (1 - frac)):.1f} s  ·  hand at {hand:.0f} mm")
                bar_fg.set_width(0.70 * frac); bar_fg.set_color(WAIT); bar_fg.set_visible(True)
            else:
                if len(votes) < 3:
                    status.set_text("RECOGNIZING"); big.set_text("…"); big.set_color(FG); sub.set_text("")
                else:
                    w, c = Counter(votes).most_common(1)[0]
                    agree = c / len(votes)
                    status.set_text("RESULT")
                    big.set_text(CLASSES[w]); big.set_color(OK if agree >= 0.8 else WAIT)
                    sub.set_text(f"{agree*100:.0f}% of last {len(votes)} frames agree"
                                 + (f"  ·  hand at {hand:.0f} mm" if hand else "")
                                 + (f"  ·  board = PC {st['agree']}/{st['cmp']}" if args.source == "both" else ""))
                    bar_fg.set_width(0.70 * agree); bar_fg.set_color(OK if agree >= 0.8 else WAIT); bar_fg.set_visible(True)
            marker.set_data([min(max(hand, 100), 450)] if hand else [], [0.82] if hand else [])
            probs = st["probs"]
            top = int(probs.argmax()) if probs.any() else -1
            for i, b in enumerate(bars):
                b.set_width(probs[i] * 100)
                b.set_color(OK if i == top else "#3A4552")
                pct_txt[i].set_text(f"{probs[i]*100:.0f}%" if probs.any() else "")
            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_draw = now
        else:
            time.sleep(0.004)
        if max_seconds and now - t0 > max_seconds:
            break
    if save_png:
        fig.savefig(save_png, dpi=80, facecolor=BG)
    ser.close()
    plt.close(fig)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด
#  1. "NO DATA" — COM3 ถูกโปรแกรมอื่นเปิดอยู่ (PuTTY) หรือพอร์ตผิด
#  2. "4×4 MODE" — เฟิร์มแวร์ไม่ใช่ 8×8 → ต้อง build/flash ใหม่
#  3. ค้างที่ PLACE HAND ทั้งที่มีมือ — status อาจถูก remap (ต้องเป็นค่าดิบ 5/9, MY_TOF_FAST_READ = 1)
#  4. ผลขึ้นหลังมือนิ่งในช่วง 1 วิ + เก็บโหวตอีก ~0.2 วิ; เปลี่ยนท่า = เอามือออกแล้วเข้าใหม่
#  5. หน้าต่างอาจอัปเดตช้ากว่าเซ็นเซอร์บนเครื่องช้า — การทายยังใช้ทุกเฟรม แค่ภาพวาดไม่ทุกเฟรม
#  6. (v3) --source board ขึ้น NO DATA ทั้งที่บอร์ดส่งข้อมูล = เฟิร์มแวร์ไม่ได้พิมพ์บรรทัด Y (MY_TOF_PRED_LINE = 0)
#  7. (v3) โหมด board: การโหวต/ช่วงระยะ 220–320/ถือ 1 วิ ทำบน PC เหมือนโหมด pc — ส่วนที่ย้ายไปบอร์ดคือ "การทาย" รายเฟรม
#     (บอร์ดยังมีโหวตของตัวเองในบรรทัด P/VOTE ซึ่งไม่มีช่วง 220–320 และไม่ต้องถือนิ่ง)
# ─────────────────────────────────────────────────────────────────────────
