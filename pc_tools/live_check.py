"""live_check.py — ลองสดกับเซ็นเซอร์จริง: วัดเวลาต่อเฟรม + ตรวจว่าบอร์ดทายตรงกับ PC ทุกเฟรม (PROJECT_FACTS §5.1y-S กติกา A17 ข้อ 5, ขั้น 5.7)

เฟิร์มแวร์ต้องเป็น MY_TOF_REPLAY = 0 (อ่านเซ็นเซอร์จริง), MY_TOF_PRED_LINE = 1, MY_TOF_FRAME_TIMING = 1
สคริปต์อ่าน UART ต่อเนื่อง --seconds วินาที ระหว่างนั้นทำท่ามือเหนือเซ็นเซอร์ไปเรื่อยๆ (ท่าไหนก็ได้ ไม่ได้วัดความแม่นยำ)
แล้วตรวจ 2 เรื่อง:
  1. เวลา : ทุกเฟรมที่ทาย บอร์ดพิมพ์ Q,<เฟรม>,sensor,preprocess,inference,decision,total (µs, DWT, HSE)
            → ค่าสูงสุดของ total ต้อง ≤ 33,000 µs (§5.1l) ; รายงาน n, mean ± SD, p99, max
  2. ตรงกับ PC : เฟรมดิบ F/S/G ที่บอร์ดพิมพ์ → st_dataset (เตรียมข้อมูลแบบเดียวกับตอนเทรน) → recentre (ถ้า E5R) → โมเดลบน PC
            → ท่าที่ทาย (หรือ -1 = นอกช่วง) ต้องตรงกับบรรทัด Y ของบอร์ด "ทุกเฟรม"
ไม่ใช่การวัดความแม่นยำ (ไม่มีป้ายท่า) และไม่ใช่ชุดข้อมูล — ไฟล์ที่บันทึกไว้ใช้ตรวจย้อนหลังเท่านั้น

วิธีรัน (ที่ root ของ repo, ปิด PuTTY ก่อน):
    python pc_tools\\live_check.py --port COM3 --model e5r --seconds 90
ผลลัพธ์: results_live\\<model>_<วันเวลา>\\  report.txt, summary.json, per_frame.csv, board_lines.txt, live_frames.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (import_st_dataset / in_range_mask / sha256_file / CLASSES)
import recentre as rc               # noqa: E402

CLASSES = e56.CLASSES
BUDGET_US = 33000                   # §5.1l: งานต่อเฟรม ≤ ครึ่งหนึ่งของคาบ 66 ms
PROB_TOL = 2e-5                     # บอร์ดเทียบ PC (float32 คนละเครื่อง) — เกณฑ์เดียวกับ replay_board.py
DEFAULT_KERAS = {"e5": Path("results_phase55_final/deploy/E5.keras"),
                 "e5r": Path("results_phase55c_final/deploy/E5R.keras")}
MIN_TIMED_FRAMES = 300              # เฟรมที่ทายอย่างน้อยเท่านี้ถึงจะสรุปเรื่องเวลา


class SerialLink:
    """ห่อ pyserial: readline(timeout) -> str ('' = หมดเวลา)"""

    def __init__(self, port: str, baud: int) -> None:
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("pyserial is required: pip install pyserial") from exc
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.ser.reset_input_buffer()
        self.buf = b""

    def readline(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buf += chunk
            elif time.monotonic() > deadline:
                return ""
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()

    def close(self) -> None:
        self.ser.close()


def capture(link, model: str, seconds: float, boot_timeout: float, log: list[str], clock=time.monotonic) -> tuple[dict, dict]:
    """รอบรรทัดบูต (ต้องกด RESET) แล้วเก็บบรรทัด F/S/G/Y/Q ต่อเฟรม --seconds วินาที"""
    boot: dict = {}
    deadline = clock() + boot_timeout
    while clock() < deadline:
        line = link.readline(0.5)
        if not line:
            continue
        log.append(line)
        if line.startswith("MY_TOF: REPLAY mode"):
            raise SystemExit("the board runs the REPLAY build — flash the MY_TOF_REPLAY = 0 build for a live check. Stop.")
        if line.startswith("MY_AI: model="):
            kv = dict(p.split("=", 1) for p in line[len("MY_AI: "):].split() if "=" in p)
            boot.update(model=kv.get("model"), recentre=kv.get("recentre"), check=kv.get("check"))
            break
    if not boot:
        raise SystemExit("no 'MY_AI: model=' boot line — press RESET when asked (close PuTTY first). Stop.")
    if boot["model"] != model or boot["check"] != "OK" or boot["recentre"] != ("1" if model == "e5r" else "0"):
        raise SystemExit(f"board says model={boot['model']} recentre={boot['recentre']} check={boot['check']} — "
                         f"does not fit --model {model}. Stop.")
    frames: dict = {}
    t_end = clock() + seconds
    last_print = 0.0
    while clock() < t_end:
        line = link.readline(0.5)
        if not line:
            continue
        log.append(line)
        parts = line.split(",")
        if len(parts) < 3 or parts[0] not in ("F", "S", "G", "Y", "Q"):
            continue
        try:
            fno = int(parts[1])
            vals = [float(v) for v in parts[2:]]
        except ValueError:
            continue
        slot = frames.setdefault(fno, {})
        if parts[0] in "FSG" and len(vals) == 64:
            slot[parts[0]] = np.array(vals)
        elif parts[0] == "Y":
            slot["y"] = int(vals[0])
            slot["p"] = np.array(vals[1:]) if len(vals) == 8 else None
        elif parts[0] == "Q" and len(vals) == 5:
            slot["q"] = [int(v) for v in vals]
        now = clock()
        if now - last_print > 5:
            n_inf = sum("q" in f for f in frames.values())
            print(f"  {t_end - now:5.0f} s left · frames {len(frames)} · inferred {n_inf}")
            last_print = now
    return boot, frames


def main(argv: list[str] | None = None, link=None, load_model: Callable | None = None, clock=time.monotonic) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--model", choices=["e5", "e5r"], required=True)
    ap.add_argument("--keras", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--boot-timeout-s", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    keras = args.keras or DEFAULT_KERAS[args.model]
    if not keras.is_file():
        raise SystemExit(f"model not found: {keras}")
    st = e56.import_st_dataset()
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)
    predict = load_model(keras)               # โหลดก่อนเปิดพอร์ต ไม่ให้เวลาโหลด TF ไปกินเวลาเก็บ

    log: list[str] = []
    own = link is None
    if own:
        link = SerialLink(args.port, args.baud)
        print("\n  >>> Press the black RESET button on the board now <<<\n")
    try:
        boot, frames = capture(link, args.model, args.seconds, args.boot_timeout_s, log, clock)
    finally:
        if own:
            link.close()
    print(f"  board: model={boot['model']} recentre={boot['recentre']} check={boot['check']}")

    # ---- เฟรมที่ได้ครบ F/S/G/Y (ตัดเฟรมแรก/สุดท้ายที่อาจขาดเพราะเริ่ม/หยุดกลางเฟรม) ----
    nos = sorted(k for k, f in frames.items() if all(x in f for x in ("F", "S", "G", "y")))
    incomplete = len(frames) - len(nos)
    if not nos:
        raise SystemExit("no complete frames received — check the firmware (PRED_LINE = 1, TIMING_MODE = 0)")
    dist = np.stack([frames[k]["F"] for k in nos], axis=1)
    status = np.stack([frames[k]["S"] for k in nos], axis=1)
    sig = np.stack([frames[k]["G"] for k in nos], axis=1)
    zone = np.zeros((4, 64, len(nos)))
    zone[st.CH_TARGET_STATUS], zone[st.CH_DISTANCE], zone[st.CH_SIGNAL] = status, dist, sig
    zone[st.CH_VALID] = np.isin(status, st.VALID_STATUS)
    mask = e56.in_range_mask(zone, st)
    X, _ = st._preprocess_session(zone, np.zeros((1, len(nos))), st.MAX_DISTANCE_MM, st.MIN_DISTANCE_MM,
                                  st.BACKGROUND_DISTANCE_MM)
    X = X.astype(np.float32)
    if len(X) != int(mask.sum()):
        raise SystemExit("in-range mask and st_dataset output differ — stop")
    P = np.asarray(predict(rc.recentre(X) if args.model == "e5r" else X)) if len(X) else np.zeros((0, 7))
    pc_cls = np.full(len(nos), -1, dtype=int)
    pc_cls[mask] = P.argmax(axis=1)
    pc_p = np.full((len(nos), 7), np.nan)
    pc_p[mask] = P

    rows = []
    for i, k in enumerate(nos):
        f = frames[k]
        pdiff = float(np.max(np.abs(f["p"] - pc_p[i]))) if f.get("p") is not None and mask[i] else float("nan")
        q = f.get("q")
        rows.append({"frame": k, "board_class": f["y"], "pc_class": int(pc_cls[i]), "prob_maxdiff": pdiff,
                     "sensor_us": q[0] if q else "", "preprocess_us": q[1] if q else "", "inference_us": q[2] if q else "",
                     "decision_us": q[3] if q else "", "total_us": q[4] if q else ""})
    agree = sum(r["board_class"] == r["pc_class"] for r in rows)
    guard = sum(r["board_class"] in (-2, -3) for r in rows)
    tot = np.array([r["total_us"] for r in rows if r["total_us"] != ""], dtype=float)
    parts = {k: np.array([r[k] for r in rows if r[k] != ""], dtype=float)
             for k in ("sensor_us", "preprocess_us", "inference_us", "decision_us")}
    pd = np.array([r["prob_maxdiff"] for r in rows], dtype=float)
    pd_max = float(np.nanmax(pd)) if np.isfinite(pd).any() else float("nan")
    gaps = np.diff(np.array(nos))
    missing = int((gaps[gaps > 1] - 1).sum()) if len(gaps) else 0
    stat = lambda a: {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),  # noqa: E731
                      "p99": float(np.percentile(a, 99)), "max": float(a.max())} if len(a) else {"n": 0}
    checks = {"class_all_frames": agree == len(rows), "no_guard_or_run_errors": guard == 0,
              "probabilities_within_tol": bool(np.isfinite(pd_max) and pd_max <= PROB_TOL),
              "enough_timed_frames": len(tot) >= MIN_TIMED_FRAMES,
              "max_total_within_budget": bool(len(tot) and tot.max() <= BUDGET_US)}
    verdict = "PASS" if all(checks.values()) else "FAIL"

    out = args.out_dir or Path("results_live") / f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "board_lines.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    np.savez(out / "live_frames.npz", zone_data=zone, frame_nos=np.array(nos), board_class=np.array([r["board_class"] for r in rows]))
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "verdict": verdict, "checks": checks, "boot": boot,
               "model": args.model, "keras": str(keras), "keras_sha256": e56.sha256_file(keras), "seconds": args.seconds,
               "frames_complete": len(rows), "frames_incomplete_dropped": incomplete, "frames_missing_in_sequence": missing,
               "in_range_frames": int(mask.sum()), "class_agree": agree, "guard_or_run_errors": guard, "prob_maxdiff": pd_max,
               "timing_us": {"total": stat(tot), **{k: stat(v) for k, v in parts.items()}}, "budget_us": BUDGET_US,
               "board_class_counts": {("none" if c < 0 else CLASSES[c]): int(sum(r["board_class"] == c for r in rows))
                                      for c in sorted({r["board_class"] for r in rows})},
               "clock": "DWT cycle counter, SystemCoreClock 84 MHz from HSE (§3.1)", "note": "live parity + timing, not accuracy"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    yes = lambda b: "PASS" if b else "FAIL"   # noqa: E731
    ts = summary["timing_us"]["total"]
    L = ["=" * 82, f"  Live check · board ({args.model}) with the real sensor · {args.seconds:g} s · FACTS §5.1y-S rule 5", "=" * 82,
         f"  frames {len(rows):,} complete (dropped incomplete {incomplete}, missing in sequence {missing}) · in range "
         f"{int(mask.sum()):,}",
         f"  {yes(checks['class_all_frames'])}  board class (or -1) = PC (st_dataset → "
         f"{'recentre → ' if args.model == 'e5r' else ''}{keras.name}) on every frame   {agree:,}/{len(rows):,}",
         f"  {yes(checks['no_guard_or_run_errors'])}  no -3 / -2 answers   {guard}",
         f"  {yes(checks['probabilities_within_tol'])}  max |board p − PC p| = {pd_max:.2e} (tolerance {PROB_TOL:.0e})"]
    if ts.get("n"):
        L += [f"  {yes(checks['enough_timed_frames'])}  timed frames (hand in range) n = {ts['n']:,} (need ≥ {MIN_TIMED_FRAMES})",
              f"  {yes(checks['max_total_within_budget'])}  per-frame work total: mean {ts['mean'] / 1000:.3f} ± {ts['sd'] / 1000:.3f} ms, "
              f"p99 {ts['p99'] / 1000:.3f}, max {ts['max'] / 1000:.3f} ms (budget {BUDGET_US / 1000:.0f} ms)"]
        for k in ("sensor_us", "preprocess_us", "inference_us", "decision_us"):
            s = summary["timing_us"][k]
            L.append(f"        {k[:-3]:<11} mean {s['mean'] / 1000:7.3f} ms   max {s['max'] / 1000:7.3f} ms")
    else:
        L.append("  FAIL  no Q lines — was the hand in 100–400 mm? is MY_TOF_FRAME_TIMING = 1?")
    L += [f"  VERDICT: {verdict}", "  board answers: " + ", ".join(f"{k} {v}" for k, v in summary["board_class_counts"].items()),
          "  clock: DWT, 84 MHz HSE · not an accuracy test (no labels)", f"  output: {out}", "=" * 82]
    text = "\n".join(L)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return summary


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ลืมปิด PuTTY → เปิด COM3 ไม่ได้ ; ไม่กด RESET → ไม่เห็นบรรทัดบูต → หยุด (ตั้งใจ: ต้องรู้ว่าบอร์ดรันโมเดลอะไร)
#  2. แฟลชตัว REPLAY ค้างไว้ → หยุดพร้อมบอก
#  3. มือไม่อยู่ในช่วง 100–400 มม. นานพอ → เฟรมที่จับเวลาน้อยกว่า 300 → FAIL ด้วยเหตุนั้น (ทำใหม่ ยื่นมือนานขึ้น)
#  4. PC อ่าน serial ไม่ทัน (เครื่องช้ามาก) → เฟรมขาดตอน (missing) — ไม่กระทบการเทียบเฟรมที่ได้ครบ
#  5. ผลนี้ไม่ใช่ความแม่นยำ — ความแม่นยำบนบอร์ดมาจาก replay (§5.1y-U) และข้อมูลเพื่อนตอนท้าย
# ─────────────────────────────────────────────────────────────────────────
