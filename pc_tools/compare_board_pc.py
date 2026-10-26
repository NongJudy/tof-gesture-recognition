"""compare_board_pc.py — Phase 5.5b ขั้น 2.4 ข้อ 7: บอร์ดทายตรงกับ PC ทีละเฟรมไหม (เกณฑ์ §5.1l = 200/200)

วิธีทดสอบ (วิธี B):
  บอร์ดอ่านมือจริง → เตรียมข้อมูลในเฟิร์มแวร์ → ทายด้วย E5 บนชิป → พิมพ์ F/S/G (ข้อมูลดิบ) + Y (ผลทาย)
  PC เอาข้อมูลดิบ "เฟรมเดียวกัน" ไปเตรียมข้อมูลด้วยฟังก์ชันที่ใช้ตอนฝึกจริง (st_dataset._preprocess_session)
  แล้วทายด้วย E5 บน PC → เทียบกันทีละเฟรม
  จึงตรวจทั้ง 3 อย่างพร้อมกัน: การตัดสินว่ามีมือในช่วง 100–400 มม. · การเตรียมข้อมูล · การทายบนชิป

เกณฑ์ผ่าน (ตั้งไว้ก่อนวัด 26 ต.ค. 2026):
  1. เฟรมที่เทียบได้ (ทั้งบอร์ดและ PC ทาย) ≥ 200 และท่าที่ทายตรงกันทุกเฟรม
  2. การตัดสิน "ทาย / ไม่ทาย" (ในช่วง / นอกช่วง) ตรงกันทุกเฟรมที่เก็บได้
  ความต่างของความน่าจะเป็นสูงสุด รายงานไว้เป็นข้อมูล (บอร์ดพิมพ์ทศนิยม 6 ตำแหน่ง)

ต้องใช้เฟิร์มแวร์ที่ตั้ง MY_TOF_PRED_LINE = 1 (บรรทัด MY_AI ตอนบูตต้องมี pred_line=1)
ข้อมูลที่เก็บเป็นการเทียบ "บอร์ด กับ PC" ไม่ใช่ความแม่นยำ — ห้ามรายงานเป็น accuracy

วิธีรัน (ที่ root ของ repo, ปิด PuTTY / live_demo ก่อน):
    python pc_tools\\compare_board_pc.py --port COM3 --model results_phase55_final\\deploy\\E5.keras
ตัวเลือก: --per-posture 29 (7 × 29 = 203 เฟรม) --out-dir <โฟลเดอร์ผลลัพธ์>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]  # ลำดับของ train_st_cnn2d.py และ my_tof.c
N_ZONES = 64
MIN_COMPARED = 200          # เกณฑ์ §5.1l
BOOT_WAIT_S = 20.0          # รอบรรทัด MY_AI หลังกด RESET
POSTURE_TIMEOUT_S = 90.0    # เก็บท่าเดียวนานเกินนี้ → ถามว่าจะลองใหม่หรือข้าม


def import_st_dataset():
    """โหลด st_dataset.py (อยู่ข้างสคริปต์นี้ใน pc_tools) — ใช้ฟังก์ชันเตรียมข้อมูลตัวเดียวกับตอนฝึก"""
    here = Path(__file__).resolve().parent
    for d in (here, here.parent):
        if (d / "st_dataset.py").is_file():
            sys.path.insert(0, str(d))
            import st_dataset  # noqa: E402  (import หลังเพิ่ม path)
            return st_dataset
    raise SystemExit("st_dataset.py not found next to this script (pc_tools)")


def sha256_file(path: Path) -> str:
    """sha256 ของไฟล์ — จดไว้ในรายงานว่าใช้โมเดลไฟล์ไหน"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_line(line: str) -> tuple[str, int, list[str]] | None:
    """'F,12,...' 'S,..' 'G,..' 'Y,12,3,p0..p6' → (ชนิด, เลขเฟรม, ค่าที่เหลือเป็นข้อความ) หรือ None"""
    parts = line.strip().split(",")
    if len(parts) < 3 or parts[0] not in ("F", "S", "G", "Y"):
        return None
    try:
        return parts[0], int(parts[1]), parts[2:]
    except ValueError:
        return None


class FrameAssembler:
    """รวมบรรทัด F/S/G/Y ที่มีเลขเฟรมเดียวกันเป็น 1 เฟรม (บอร์ดพิมพ์ F,S,G ก่อนแล้วตามด้วย Y)"""

    def __init__(self) -> None:
        self.pending: dict[int, dict[str, list[str]]] = {}
        self.incomplete_dropped = 0

    def add(self, kind: str, fno: int, vals: list[str]) -> dict | None:
        """เพิ่ม 1 บรรทัด ถ้าเฟรมนั้นครบ 4 บรรทัดแล้วคืน dict ของเฟรม"""
        slot = self.pending.setdefault(fno, {})
        slot[kind] = vals
        done = None
        if all(k in slot for k in "FSGY"):
            done = self._build(fno, self.pending.pop(fno))
        # เฟรมเก่าที่ไม่ครบ (เช่นบรรทัดหายระหว่างล้าง buffer) ทิ้งและนับไว้
        for k in [k for k in self.pending if fno - k > 5]:
            del self.pending[k]
            self.incomplete_dropped += 1
        return done

    @staticmethod
    def _build(fno: int, slot: dict[str, list[str]]) -> dict | None:
        try:
            dist = np.array([float(v) for v in slot["F"]])
            status = np.array([float(v) for v in slot["S"]])
            sig = np.array([float(v) for v in slot["G"]])
            ycls = int(slot["Y"][0])
            yprob = np.array([float(v) for v in slot["Y"][1:]]) if ycls >= 0 else None
        except (ValueError, IndexError):
            return None
        if not (len(dist) == len(status) == len(sig) == N_ZONES):
            return None
        if yprob is not None and len(yprob) != len(CLASSES):
            return None
        return {"frame": fno, "dist": dist, "status": status, "sig": sig, "board_cls": ycls, "board_p": yprob}


def pc_preprocess(st, dist: np.ndarray, status: np.ndarray, sig: np.ndarray) -> np.ndarray | None:
    """เตรียมข้อมูล 1 เฟรมด้วย st_dataset._preprocess_session (ตัวเดียวกับตอนฝึก)

    คืน x (8, 8, 2) float32 หรือ None ถ้ามือนอกช่วง 100–400 มม. (ฟังก์ชันนี้ทิ้งเฟรมนั้น)
    """
    zone = np.zeros((4, N_ZONES, 1))
    zone[st.CH_TARGET_STATUS, :, 0] = status
    zone[st.CH_VALID, :, 0] = 1.0            # ช่อง valid ของ npz ไม่ถูกใช้ในฟังก์ชันนี้ (ใช้ status แทน)
    zone[st.CH_SIGNAL, :, 0] = sig
    zone[st.CH_DISTANCE, :, 0] = dist
    X, _ = st._preprocess_session(zone, np.zeros((1, 1)), st.MAX_DISTANCE_MM, st.MIN_DISTANCE_MM,
                                  st.BACKGROUND_DISTANCE_MM)
    return X[0].astype(np.float32) if len(X) == 1 else None


def read_lines(ser, buf: bytearray) -> list[str]:
    """ดึงทุกบรรทัดที่มาครบแล้วจาก serial (ไม่ค้าง)"""
    n = ser.in_waiting
    if n:
        buf += ser.read(n)
    *lines, rest = bytes(buf).split(b"\n")
    buf[:] = rest
    return [ln.decode("utf-8", errors="ignore").strip() for ln in lines]


def wait_boot(ser, raw_log, wait_s: float, clock: Callable[[], float]) -> list[str]:
    """รอบรรทัดตอนบูต (MY_TOF / CLK / MY_AI) หลังผู้ใช้กด RESET คืนบรรทัดที่เจอ"""
    buf = bytearray()
    found: list[str] = []
    t_end = clock() + wait_s
    while clock() < t_end:
        for ln in read_lines(ser, buf):
            raw_log.write(ln + "\n")
            if ln.startswith(("T1 ", "MY_TOF:", "CLK,", "MY_AI:")):
                found.append(ln)
                print("   board:", ln)
            if ln.startswith("MY_AI:"):
                return found
        time.sleep(0.02)
    return found


def collect_posture(ser, raw_log, asm: FrameAssembler, k: int, timeout_s: float,
                    clock: Callable[[], float]) -> tuple[list[dict], bool]:
    """เก็บเฟรมจนได้ k เฟรมที่บอร์ด "ทาย" (Y >= 0) — เฟรม -1 ก็เก็บไว้ใช้ตรวจการตัดสินในช่วง/นอกช่วง"""
    buf = bytearray()
    ser.reset_input_buffer()
    frames: list[dict] = []
    n_inf = 0
    t_end = clock() + timeout_s
    last_print = 0.0
    while clock() < t_end:
        for ln in read_lines(ser, buf):
            raw_log.write(ln + "\n")
            p = parse_line(ln)
            if p is None:
                continue
            fr = asm.add(*p)
            if fr is None:
                continue
            frames.append(fr)
            if fr["board_cls"] >= 0:
                n_inf += 1
            if n_inf >= k:
                print(f"\r   collected {n_inf}/{k} predicted frames (+{len(frames) - n_inf} not predicted)   ")
                return frames, True
        if clock() - last_print > 0.25:
            print(f"\r   collected {n_inf}/{k} predicted frames ...", end="", flush=True)
            last_print = clock()
        time.sleep(0.01)
    print()
    return frames, False


def compare(frames: list[dict], st, predict: Callable[[np.ndarray], np.ndarray]) -> dict:
    """ฝั่ง PC: เตรียมข้อมูล + ทายทีละเฟรม แล้วเทียบกับบอร์ด"""
    xs, idx = [], []
    for i, fr in enumerate(frames):
        x = pc_preprocess(st, fr["dist"], fr["status"], fr["sig"])
        fr["pc_in_range"] = x is not None
        if x is not None:
            xs.append(x)
            idx.append(i)
    if xs:
        probs = predict(np.stack(xs))
        for i, p in zip(idx, probs):
            frames[i]["pc_p"] = np.asarray(p, dtype=np.float64)
            frames[i]["pc_cls"] = int(np.argmax(p))
    gate_mismatch, cls_mismatch, diffs, n_compared, board_errors = [], [], [], 0, []
    for fr in frames:
        if fr["board_cls"] == -2:
            board_errors.append(fr["frame"])
            continue
        board_in = fr["board_cls"] >= 0
        if board_in != fr["pc_in_range"]:
            gate_mismatch.append(fr["frame"])
            continue
        if not board_in:
            continue
        n_compared += 1
        d = float(np.abs(fr["board_p"] - fr["pc_p"]).max())
        fr["max_abs_diff"] = d
        diffs.append(d)
        if fr["board_cls"] != fr["pc_cls"]:
            cls_mismatch.append(fr["frame"])
    return {"n_frames": len(frames), "n_compared": n_compared, "gate_mismatch": gate_mismatch,
            "cls_mismatch": cls_mismatch, "board_errors": board_errors,
            "max_abs_diff": max(diffs) if diffs else float("nan"),
            "median_abs_diff": float(np.median(diffs)) if diffs else float("nan")}


def write_outputs(out_dir: Path, frames: list[dict], res: dict, meta: dict) -> str:
    """frames.csv + report.txt — คืนข้อความรายงาน"""
    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "prompted_posture", "board_cls", "pc_in_range", "pc_cls", "max_abs_diff"]
                   + [f"board_p{i}" for i in range(7)] + [f"pc_p{i}" for i in range(7)])
        for fr in frames:
            bp = fr["board_p"] if fr["board_p"] is not None else [""] * 7
            pp = fr.get("pc_p", [""] * 7)
            w.writerow([fr["frame"], fr["prompt"], fr["board_cls"], int(fr["pc_in_range"]), fr.get("pc_cls", ""),
                        fr.get("max_abs_diff", "")] + list(bp) + list(pp))

    passed = (res["n_compared"] >= MIN_COMPARED and not res["cls_mismatch"] and not res["gate_mismatch"]
              and not res["board_errors"])
    L = ["=" * 66, "  Board vs PC agreement (Phase 5.5b step 2.4 item 7, rule §5.1l)", "=" * 66,
         f"  run at          {meta['run_at']}",
         f"  model           {meta['model']}  sha256 {meta['model_sha'][:16]}...",
         f"  boot lines      {' | '.join(meta['boot']) if meta['boot'] else 'NOT SEEN (board not reset?)'}",
         f"  frames logged   {res['n_frames']}  (incomplete frames dropped: {meta['incomplete']})",
         f"  compared        {res['n_compared']}  (both board and PC predicted)",
         f"  class agree     {res['n_compared'] - len(res['cls_mismatch'])} / {res['n_compared']}",
         f"  gate mismatch   {len(res['gate_mismatch'])}  (board vs PC disagree on 'hand in 100-400 mm')",
         f"  board run error {len(res['board_errors'])}",
         f"  prob |diff|     max {res['max_abs_diff']:.2e}   median {res['median_abs_diff']:.2e}"
         "   (board prints 6 decimals)",
         "", "  Board-predicted class per prompted posture (NOT accuracy — agreement test only):"]
    for c in CLASSES:
        got = [fr["board_cls"] for fr in frames if fr["prompt"] == c and fr["board_cls"] >= 0]
        cnt = {CLASSES[j]: got.count(j) for j in range(7) if got.count(j)}
        L.append(f"    {c:<11} n={len(got):>3}  {cnt}")
    if res["cls_mismatch"]:
        L.append(f"  first class mismatches (frame): {res['cls_mismatch'][:10]}")
    if res["gate_mismatch"]:
        L.append(f"  first gate mismatches (frame):  {res['gate_mismatch'][:10]}")
    L += ["", f"  RESULT: {'PASS' if passed else 'FAIL'}  (need >= {MIN_COMPARED} compared, 0 class / gate / run errors)",
          f"  output: {out_dir}", "=" * 66]
    text = "\n".join(L)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str] | None = None, serial_factory=None, predictor=None,
         ask: Callable[[str], str] = input, clock: Callable[[], float] = time.monotonic) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--model", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--per-posture", type=int, default=29, help="predicted frames per posture (7 x 29 = 203)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    st = import_st_dataset()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    model_sha = sha256_file(args.model)
    if predictor is None:
        import tensorflow as tf  # โหลดช้า จึงโหลดเมื่อใช้จริงเท่านั้น
        model = tf.keras.models.load_model(str(args.model), compile=False)
        if model.output_shape[-1] != len(CLASSES):
            raise SystemExit(f"model has {model.output_shape[-1]} outputs, expected 7")
        predictor = lambda X: model.predict(X, batch_size=256, verbose=0)  # noqa: E731
    if serial_factory is None:
        import serial
        serial_factory = lambda: serial.Serial(args.port, args.baud, timeout=0.05)  # noqa: E731

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("results_phase55_final") / "board_check_live" / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with serial_factory() as ser, open(out_dir / "raw_serial.log", "w", encoding="utf-8") as raw_log:
        print("\nPress the black RESET button on the board now "
              f"(waiting up to {BOOT_WAIT_S:.0f} s for the MY_AI line) ...")
        boot = wait_boot(ser, raw_log, BOOT_WAIT_S, clock)
        ai = next((b for b in boot if b.startswith("MY_AI:")), "")
        if ai and "pred_line=1" not in ai:
            raise SystemExit("Board firmware does not print Y lines (pred_line=1 missing). "
                             "Set MY_TOF_PRED_LINE = 1, build and flash first.")
        if not ai:
            print("   [WARNING] MY_AI line not seen - continuing; the report will say so.")

        asm = FrameAssembler()
        frames: list[dict] = []
        for n, c in enumerate(CLASSES, 1):
            while True:
                ask(f"\n[{n}/7] Posture: {c}. Hold it over the sensor (about 27 cm), then press Enter ")
                got, ok = collect_posture(ser, raw_log, asm, args.per_posture, POSTURE_TIMEOUT_S, clock)
                for fr in got:
                    fr["prompt"] = c
                if ok:
                    frames += got
                    break
                ans = ask(f"   Not enough predicted frames in {POSTURE_TIMEOUT_S:.0f} s. "
                          "r = retry this posture, s = skip it: ").strip().lower()
                if ans != "r":
                    frames += got
                    break

    print("\nComputing PC predictions (same preprocessing as training: st_dataset) ...")
    res = compare(frames, st, predictor)
    meta = {"run_at": stamp, "model": str(args.model), "model_sha": model_sha, "boot": boot,
            "incomplete": asm.incomplete_dropped}
    print("\n" + write_outputs(out_dir, frames, res, meta))
    return res


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ลืมแฟลชเฟิร์มแวร์ที่ MY_TOF_PRED_LINE = 1 → สคริปต์หยุดเองถ้าเห็นบรรทัด MY_AI ไม่มี pred_line=1;
#     ถ้าไม่ได้กด RESET จะไม่เห็นบรรทัดนี้ และจะเก็บไม่ได้สักเฟรม (ไม่มีบรรทัด Y) → ครบเวลาแล้วถาม r/s
#  2. COM3 ถูกโปรแกรมอื่นจับอยู่ (PuTTY, live_demo) → เปิดพอร์ตไม่ได้ ปิดโปรแกรมนั้นก่อน (TRAP #19)
#  3. เฟรมที่ไม่ครบ 4 บรรทัด (หายตอนล้าง buffer เปลี่ยนท่า) ถูกทิ้งและนับใน "incomplete frames dropped" — ไม่นับเป็นไม่ตรงกัน
#  4. ความต่างของความน่าจะเป็นมีขั้นต่ำ ~5e-7 เพราะบอร์ดพิมพ์ทศนิยม 6 ตำแหน่ง — ไม่ใช่ความคลาดเคลื่อนของชิป
#  5. ผลนี้คือ "บอร์ดทายเหมือน PC" ไม่ใช่ "ทายถูก" — ตารางท่าที่บอร์ดทายต่อท่าที่ขอให้ทำ ใช้ดูความครอบคลุมเท่านั้น
#  6. ถ้า class ไม่ตรงในเฟรมที่ความน่าจะเป็น 2 อันดับแรกเกือบเท่ากัน (ต่างกัน < 1e-5) ให้ดูตัวเลขใน frames.csv
#     ก่อนสรุปว่าเฟิร์มแวร์ผิด — แต่ห้ามเปลี่ยนเกณฑ์ผ่านหลังเห็นผล (บันทึกเป็นข้อสังเกตแทน)
# ─────────────────────────────────────────────────────────────────────────
