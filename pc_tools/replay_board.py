"""replay_board.py — ส่งเฟรมที่บันทึกไว้เข้าบอร์ด (เฟิร์มแวร์ MY_TOF_REPLAY = 1) แล้วเทียบผลทายของบอร์ดทุกเฟรม
(PROJECT_FACTS §5.1y-S กติกา A17 ข้อ 5 "board = PC ทุกเฟรม", ขั้น 5.5–5.6)

ทำไมต้อง replay: บอร์ดใช้แค่ ระยะ (uint16) / status (uint8) / signal (uint32) ต่อโซน และไฟล์ .npz ที่เก็บไว้มีค่าเหล่านี้
ครบเป็นจำนวนเต็ม (ได้จากบรรทัด F/S/G ของบอร์ดเอง) → ส่งกลับเข้าไปได้ "ตรงทุกบิต" บอร์ดจึงประมวลผลเหมือนตอนวัดจริง
โดยไม่ต้องเก็บข้อมูลใหม่ (§5.1z)

สองโหมดอ้างอิง (--reference):
  board : เทียบกับบรรทัด Y ที่บอร์ดพิมพ์ไว้ตอนเก็บข้อมูล (*_board.csv) — ใช้กับ E5 เท่านั้น
          = ทดสอบ "วิธี replay" ว่าทำให้บอร์ดทายซ้ำได้เหมือนเดิมทุกเฟรม (ขั้น 5.5)
  pc    : เทียบกับโมเดลบน PC (.keras) ตัวเดียวกับที่แปลงลงบอร์ด ; E5R จะถูก re-centre ก่อนเข้าโมเดล (ขั้น 5.6)
ทุกเฟรมถูกตรวจ 3 อย่าง: (1) บอร์ดทวนเฟรมที่ได้รับ (F/S/G) ตรงกับที่ส่งทุกค่า (2) ผลทาย (ท่า หรือ -1 = นอกช่วง) ตรงกับ
อ้างอิง (3) ความน่าจะเป็น 7 ค่าต่างจากอ้างอิงไม่เกินเกณฑ์ ; บอร์ดตอบ -3 (โมเดลกับสวิตช์ไม่เข้าคู่) หรือ -2 (โมเดลรันพลาด) = ไม่ผ่าน

วิธีรัน (ที่ root ของ repo, ปิด PuTTY ก่อน):
    ขั้น 5.5 (E5):  python pc_tools\\replay_board.py --port COM3 --model e5  --reference board --data-dir data\\phase56_stand
    ขั้น 5.6 (E5R): python pc_tools\\replay_board.py --port COM3 --model e5r --reference pc    --data-dir data\\phase56_stand
    ลองสั้นๆ ก่อน:   เพิ่ม --limit-takes 1
ผลลัพธ์: results_replay\\<model>_<ชุดข้อมูล>_<reference>_<วันเวลา>\\  report.txt, summary.json, per_frame.csv, board_lines.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (load_takes / predict_takes / sha256_file / CLASSES)
import recentre as rc               # noqa: E402

CLASSES = e56.CLASSES
SYNC = b"\xA5\x5A"
PAYLOAD = 2 + 64 * 2 + 64 + 64 * 4          # seq + ระยะ + status + signal = 450 (ต้องตรงกับ MY_RP_PAYLOAD)
FRAME_BYTES = 2 + PAYLOAD + 2               # 454 (ต้องตรงกับ MY_RP_FRAME_BYTES)
DEFAULT_KERAS = {"e5": Path("results_phase55_final/deploy/E5.keras"),
                 "e5r": Path("results_phase55c_final/deploy/E5R.keras")}
PROB_TOL = {"board": 2e-6,                  # บอร์ดเทียบบอร์ด: คำนวณเหมือนกันทุกบิต ต่างได้แค่การปัด %.6f
            "pc": 2e-5}                     # บอร์ดเทียบ PC: float32 คนละเครื่อง ลำดับการบวกต่างกัน


def encode_packet(seq: int, dist: np.ndarray, status: np.ndarray, sig: np.ndarray) -> bytes:
    """สร้างแพ็กเก็ต 454 ไบต์ (little-endian) — รูปแบบเดียวกับที่ my_tof.c (MY_TOF_REPLAY) อ่าน"""
    payload = (struct.pack("<H", seq) + struct.pack("<64H", *dist.astype(np.uint16).tolist())
               + bytes(status.astype(np.uint8).tolist()) + struct.pack("<64I", *sig.astype(np.uint32).tolist()))
    assert len(payload) == PAYLOAD
    return SYNC + payload + struct.pack("<H", sum(payload) & 0xFFFF)


def zone_ints(zone: np.ndarray, j: int, st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ดึงค่าดิบของเฟรม j จาก zone_data (4, 64, n) และตรวจว่าเป็นจำนวนเต็มในช่วงชนิดข้อมูลของบอร์ด"""
    out = []
    for ch, hi in ((st.CH_DISTANCE, 65535), (st.CH_TARGET_STATUS, 255), (st.CH_SIGNAL, 4294967295)):
        v = zone[ch, :, j]
        if not np.all(np.isfinite(v)) or np.any(v != np.round(v)) or v.min() < 0 or v.max() > hi:
            raise SystemExit(f"frame {j}: channel {ch} is not a valid integer in 0..{hi} — cannot replay it exactly")
        out.append(v.astype(np.int64))
    return out[0], out[1], out[2]


class SerialLink:
    """ห่อ pyserial ให้มี write(bytes) และ readline(timeout) -> str ('' = หมดเวลา)"""

    def __init__(self, port: str, baud: int) -> None:
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("pyserial is required: pip install pyserial") from exc
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = b""

    def write(self, data: bytes) -> None:
        self.ser.write(data)
        self.ser.flush()

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

    def reset_input(self) -> None:
        self.ser.reset_input_buffer()
        self.buf = b""

    def close(self) -> None:
        self.ser.close()


def wait_boot(link, model: str, timeout: float, log: list[str]) -> dict:
    """รอบรรทัดบูตของบอร์ด: MY_AI: model=<m> recentre=<r> check=OK และ REPLAY,ready,454"""
    info: dict = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = link.readline(0.5)
        if not line:
            continue
        log.append(line)
        if line.startswith("MY_AI: model="):
            kv = dict(p.split("=", 1) for p in line[len("MY_AI: "):].split() if "=" in p)
            info.update(model=kv.get("model"), recentre=kv.get("recentre"), check=kv.get("check"))
        elif line.startswith("MY_TOF: REPLAY mode"):
            info["replay_mode"] = True
        elif line.startswith("REPLAY,ready"):
            info["packet_bytes"] = int(line.split(",")[2])
            break
        elif line.startswith("F,"):
            raise SystemExit("the board is streaming sensor frames — it runs normal firmware, not MY_TOF_REPLAY = 1")
    if "packet_bytes" not in info:
        raise SystemExit("no 'REPLAY,ready' line — flash the MY_TOF_REPLAY = 1 build, close PuTTY, press RESET when asked")
    if info.get("model") != model:
        raise SystemExit(f"board runs model '{info.get('model')}' but --model {model} was given — wrong firmware. Stop.")
    if info.get("check") != "OK":
        raise SystemExit(f"board model guard says {info.get('check')} (recentre={info.get('recentre')}) — fix the switch. Stop.")
    if info.get("recentre") != ("1" if model == "e5r" else "0"):
        raise SystemExit(f"recentre={info.get('recentre')} does not fit model {model}. Stop.")
    if info["packet_bytes"] != FRAME_BYTES:
        raise SystemExit(f"board expects {info['packet_bytes']}-byte packets, this script sends {FRAME_BYTES}. Stop.")
    return info


def send_frame(link, seq: int, pkt: bytes, sent: tuple, timeout: float, retries: int, log: list[str]) -> dict:
    """ส่ง 1 เฟรม แล้วอ่านจนเจอ K,<seq> — คืนผลของบอร์ด (ท่า, ความน่าจะเป็น, echo ตรงไหม, เวลา, จำนวนครั้งที่ส่งซ้ำ)"""
    tag = str(seq)
    for attempt in range(retries + 1):
        link.write(pkt)
        res: dict = {"echo": {}, "y": None, "p": None, "q": None, "resend": attempt, "k": None}
        deadline = time.monotonic() + timeout
        bad = False
        while time.monotonic() < deadline:
            line = link.readline(max(0.01, deadline - time.monotonic()))
            if not line:
                break
            log.append(line)
            parts = line.split(",")
            if parts[0] in ("F", "S", "G") and len(parts) == 66 and parts[1] == tag:
                res["echo"][parts[0]] = np.array([int(v) for v in parts[2:]], dtype=np.int64)
            elif parts[0] == "Y" and len(parts) >= 3 and parts[1] == tag:
                res["y"] = int(parts[2])
                if len(parts) == 10:
                    res["p"] = np.array([float(v) for v in parts[3:]])
            elif parts[0] == "Q" and len(parts) == 7 and parts[1] == tag:
                res["q"] = [int(v) for v in parts[2:]]
            elif parts[0] == "E":
                bad = True                                  # checksum ผิด → ส่งซ้ำ
                break
            elif parts[0] == "K" and parts[1] == tag:
                res["k"] = [int(v) for v in parts[2:]]
                break
        if not bad and res["k"] is not None:
            d, s, g = sent
            res["echo_ok"] = (all(k in res["echo"] for k in "FSG") and np.array_equal(res["echo"]["F"], d)
                              and np.array_equal(res["echo"]["S"], s) and np.array_equal(res["echo"]["G"], g))
            return res
    raise SystemExit(f"frame seq {seq}: no K,{seq} after {retries + 1} tries — check cable / firmware. Stop.")


def reference(takes: list[dict], mode: str, model: str, keras: Path | None, load_model: Callable | None):
    """คำตอบอ้างอิงต่อรอบ: (ท่ายาว n [-1 = นอกช่วง, -9 = ไม่มีข้อมูล], ความน่าจะเป็น (n,7) [nan = ไม่มี])"""
    if mode == "board":
        if not all(t["board"]["present"] for t in takes):
            raise SystemExit("some takes have no *_board.csv — --reference board needs them")
        return [(t["board"]["cls"].copy(), t["board"]["p"].copy()) for t in takes]
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)
    m = load_model(keras)
    fn = (lambda X: m(rc.recentre(X))) if model == "e5r" else m     # E5R ต้อง re-centre (เหมือนตอนเทรน)
    cls, prob = e56.predict_takes(takes, fn)
    out = []
    for i, t in enumerate(takes):
        p = np.full((t["n"], len(CLASSES)), np.nan)
        p[t["mask"]] = prob[i]
        out.append((cls[i].copy(), p))
    return out


def main(argv: list[str] | None = None, link=None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", choices=["e5", "e5r"], required=True, help="โมเดลที่อยู่บนบอร์ด (ต้องตรงกับบรรทัดบูต)")
    ap.add_argument("--reference", choices=["board", "pc"], required=True)
    ap.add_argument("--keras", type=Path, default=None, help="โมเดลบน PC สำหรับ --reference pc")
    ap.add_argument("--limit-takes", type=int, default=0, help="ลองแค่ N รอบแรก (0 = ทั้งหมด)")
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--boot-timeout-s", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.reference == "board" and args.model != "e5":
        raise SystemExit("--reference board = the Y lines recorded during collection, which came from E5 — use it with --model e5")
    keras = args.keras or DEFAULT_KERAS[args.model]
    if args.reference == "pc" and not keras.is_file():
        raise SystemExit(f"model not found: {keras}")

    st = e56.import_st_dataset()
    print(f"Loading {args.data_dir} ...")
    takes = e56.load_takes(args.data_dir, st, with_board=(args.reference == "board"))
    if args.limit_takes:
        takes = takes[:args.limit_takes]
    for t in takes:
        with np.load(t["file"], allow_pickle=True) as z:
            t["zone"] = z["zone_data"].astype(np.float64)
    n_frames = sum(t["n"] for t in takes)
    print(f"  {len(takes)} takes, {n_frames} frames (all frames are sent; the board decides in/out of range itself)")
    print(f"Computing the reference ({args.reference}) ...")
    refs = reference(takes, args.reference, args.model, keras, load_model)

    log: list[str] = []
    own_link = link is None
    if own_link:
        link = SerialLink(args.port, args.baud)
        link.reset_input()
        print("\n  >>> Press the black RESET button on the board now <<<\n")
    try:
        boot = wait_boot(link, args.model, args.boot_timeout_s, log)
        print(f"  board: model={boot['model']} recentre={boot['recentre']} check={boot['check']} replay packet "
              f"{boot['packet_bytes']} B — OK")
        rows, t0, seq = [], time.monotonic(), 0
        for ti, (t, (rcls, rp)) in enumerate(zip(takes, refs)):
            for j in range(t["n"]):
                seq = seq % 65535 + 1
                d, s, g = zone_ints(t["zone"], j, st)
                res = send_frame(link, seq, encode_packet(seq, d, s, g), (d, s, g), args.timeout_s, args.retries, log)
                pdiff = (float(np.max(np.abs(res["p"] - rp[j]))) if res["p"] is not None and np.all(np.isfinite(rp[j]))
                         else float("nan"))
                rows.append({"take": f"{t['cls']} {t['take']}", "frame": j, "seq": seq, "label": t["label"],
                             "ref_class": int(rcls[j]), "board_class": res["y"], "echo_ok": res["echo_ok"],
                             "prob_maxdiff": pdiff, "resend": res["resend"],
                             "inference_us": res["q"][2] if res["q"] else "", "preprocess_us": res["q"][1] if res["q"] else "",
                             "rx_err": res["k"][2] if res["k"] and len(res["k"]) > 2 else ""})
            el = time.monotonic() - t0
            done = sum(x["n"] for x in takes[:ti + 1])
            agree = sum(r["board_class"] == r["ref_class"] for r in rows)
            print(f"  take {ti + 1:2d}/{len(takes)} {t['cls']:<11}{t['take']}  frames {done:5d}/{n_frames}  "
                  f"agree {agree}/{len(rows)}  {el / 60:4.1f} min")
    finally:
        if own_link:
            link.close()

    # ---- สรุป ----
    tol = PROB_TOL[args.reference]
    comp = [r for r in rows if r["ref_class"] != -9]
    agree = sum(r["board_class"] == r["ref_class"] for r in comp)
    echo_ok = sum(bool(r["echo_ok"]) for r in rows)
    guard = sum(r["board_class"] in (-2, -3) for r in rows)
    pd = np.array([r["prob_maxdiff"] for r in rows], dtype=float)
    pd_max = float(np.nanmax(pd)) if np.isfinite(pd).any() else float("nan")
    inr = [r for r in rows if r["board_class"] is not None and r["board_class"] >= 0]
    acc = 100.0 * sum(r["board_class"] == r["label"] for r in inr) / len(inr) if inr else float("nan")
    inf = [r["inference_us"] for r in rows if r["inference_us"] != ""]
    pre = [r["preprocess_us"] for r in rows if r["preprocess_us"] != ""]
    checks = {"echo_all_frames": echo_ok == len(rows), "class_all_frames": agree == len(comp) and len(comp) == len(rows),
              "no_guard_or_run_errors": guard == 0, "probabilities_within_tol": bool(pd_max <= tol) if np.isfinite(pd_max) else False}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    out = args.out_dir or Path("results_replay") / f"{args.model}_{args.data_dir.name}_{args.reference}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "board_lines.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "verdict": verdict, "checks": checks,
               "data_dir": str(args.data_dir), "model": args.model, "reference": args.reference,
               "keras": str(keras) if args.reference == "pc" else None,
               "keras_sha256": e56.sha256_file(keras) if args.reference == "pc" else None,
               "boot": boot, "frames": len(rows), "frames_compared": len(comp), "class_agree": agree,
               "echo_ok": echo_ok, "guard_or_run_errors": guard, "prob_maxdiff": pd_max, "prob_tol": tol,
               "resends": int(sum(r["resend"] for r in rows)), "board_accuracy_in_range_pct": acc,
               "in_range_frames": len(inr),
               "inference_us_mean_max": [float(np.mean(inf)), int(max(inf))] if inf else None,
               "preprocess_us_mean_max": [float(np.mean(pre)), int(max(pre))] if pre else None,
               "note": "replay = board computation on recorded frames; sensor time is not included (Q sensor_us = 0)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    yes = lambda b: "PASS" if b else "FAIL"   # noqa: E731
    L = ["=" * 82, f"  Replay · board ({args.model}) vs {args.reference} · {args.data_dir} · FACTS §5.1y-S rule 5", "=" * 82,
         f"  frames sent {len(rows):,} (all frames, in and out of range) · resends {summary['resends']}",
         f"  {yes(checks['echo_all_frames'])}  board echoed every frame exactly (F/S/G = sent)   {echo_ok:,}/{len(rows):,}",
         f"  {yes(checks['class_all_frames'])}  board class (or -1) = reference on every frame   {agree:,}/{len(comp):,}",
         f"  {yes(checks['no_guard_or_run_errors'])}  no -3 (model/switch guard) and no -2 (run error)  {guard}",
         f"  {yes(checks['probabilities_within_tol'])}  max |board p − reference p| = {pd_max:.2e} (tolerance {tol:.0e})",
         f"  VERDICT: {verdict}", "",
         f"  board accuracy on in-range frames (vs labels): {acc:.2f} %  (n = {len(inr):,})"]
    if inf:
        L.append(f"  board inference {np.mean(inf) / 1000:.3f} ms mean, {max(inf) / 1000:.3f} ms max · preprocess "
                 f"{np.mean(pre):.0f} µs mean, {max(pre)} µs max (replay: no sensor read in these numbers)")
    L += ["  per-frame details: per_frame.csv · every line the board sent: board_lines.txt", f"  output: {out}", "=" * 82]
    text = "\n".join(L)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return summary


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ลืมปิด PuTTY → เปิด COM3 ไม่ได้ ; ลืมกด RESET → ไม่เห็นบรรทัดบูต (รอ --boot-timeout-s แล้วหยุด)
#  2. แฟลชเฟิร์มแวร์ปกติ (REPLAY = 0) → เห็นบรรทัด F, ไหลมาเอง → หยุดพร้อมบอก
#  3. --model ไม่ตรงกับโมเดลบนบอร์ด หรือสวิตช์ RECENTRE ไม่เข้าคู่ → หยุดตั้งแต่บูต ไม่ส่งข้อมูล
#  4. checksum ผิด (สัญญาณรบกวน) → บอร์ดพิมพ์ E แล้ว PC ส่งซ้ำ (นับใน resends) ; ไม่มี K เกิน --retries → หยุด
#  5. ค่าความน่าจะเป็นแบบ pc ต่างได้เล็กน้อย (float32 คนละเครื่อง) — เกณฑ์หลักคือ "ท่าตรงทุกเฟรม"
#  6. เวลาใน Q ตอน replay ไม่มีเวลาอ่านเซ็นเซอร์ — เวลาจริงต่อเฟรมต้องวัดตอนลองสด (ขั้น 5.7)
# ─────────────────────────────────────────────────────────────────────────
