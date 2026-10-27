"""a15_decide.py — กติกา A15 (PROJECT_FACTS §5.1y): ตัดสินว่า E5R ขึ้นแทน E5 ได้ไหม สำหรับ "คนที่โมเดลรู้จัก"
ใช้ชุดนั่ง data\\phase56 เท่านั้น — ชุดยืน data\\phase56_stand ห้ามแตะจนกว่าจะตัดสินเสร็จ (สคริปต์นี้ไม่ยอมเปิด)

กติกา (เขียนไว้ก่อน E5R มีอยู่, commit a0a0ed2):
  1. ตัวกันบั๊ก : E5R (PC, ข้อมูล re-centre แล้ว) บน phase56 < 80 %  → หยุดหาบั๊กก่อน ไม่ใช่การตัดสิน
  2. ห้ามแย่ลง : E5R ≥ E5 − 3.0 pp  โดย E5 ใช้ "ผลที่บอร์ดทายเอง" (บรรทัด Y ใน *_board.csv ที่บันทึกตอนเก็บ)
  3. ผ่านข้อ 2 → E5R ไปต่อได้ แต่ยังต้องผ่าน board = PC 100 % และเวลา ≤ 33 ms ก่อนขึ้นบอร์ดจริง
     ไม่ผ่าน → E5 อยู่บนบอร์ดต่อ
  4. ไม่มีกติการายท่า (ท่าละ 2 รอบ 1 รอบขยับ recall ≈ 50 pp) — รายท่า/รายรอบ "รายงาน" อย่างเดียว

ตรวจความถูกต้องของตัวเองก่อนตัดสิน (ถ้าไม่ผ่าน = หยุด):
  ก. E5 บน PC (สคริปต์นี้คำนวณใหม่) ต้องตรงกับผลของบอร์ดในไฟล์ทุกเฟรม → ยืนยันว่าอ่านข้อมูล/เตรียมข้อมูล
     เหมือนการประเมิน 26 ต.ค. ทุกอย่าง (§5.1p: 4,219/4,219)
  ข. ความแม่นยำของ E5 บนบอร์ดต้องได้ 93.0 % ตาม §5.1p (คลาดได้ไม่เกิน 0.05 pp)
  ค. E5R ได้ข้อมูลที่ re-centre แล้วจริง (นับเฟรมที่จุดศูนย์กลางมืออยู่ใน (3, 4])

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a15_decide.py
ค่าเริ่มต้น: --data-dir data\\phase56 --ref-dir data\\phase4 --e5 results_phase55_final\\deploy\\E5.keras
            --e5r results_phase55c_final\\deploy\\E5R.keras --out-dir results_a15
ผลลัพธ์: results_a15\\  decision.json (ผลตัดสิน + sha256 ของโมเดล/ข้อมูล), report.txt, per_take.csv
รันซ้ำได้ด้วยโมเดลเดิม (ได้ผลเดิม) แต่ถ้า E5R.keras เปลี่ยนไฟล์ สคริปต์จะไม่ยอม — ตัดสินได้ครั้งเดียวต่อโมเดล
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (load_takes / predict_takes / score / board_vs_pc / centroid)
import recentre as rc               # noqa: E402

CLASSES = e56.CLASSES
BUG_GUARD_PCT = 80.0                # §5.1y ข้อ 1
MARGIN_PP = 3.0                     # §5.1y ข้อ 2
E5_BOARD_EXPECTED = 93.0            # §5.1p — ใช้ตรวจว่าสคริปต์นี้วัดแบบเดียวกับวันที่ 26 ต.ค.
E5_BOARD_TOL = 0.05
ALLOWED_DATA_DIR = "phase56"        # §5.1y: ชุดนั่งเท่านั้น


def pooled(takes: list[dict], preds: dict[int, np.ndarray]) -> dict:
    """ความแม่นยำรวมทุกเฟรมในช่วงของทุกคน (กติกาใช้ค่านี้) + โหวต 15 เฟรมแบบเฟิร์มแวร์"""
    y, p, vy, vp = [], [], [], []
    for i, t in enumerate(takes):
        m = t["mask"]
        y.append(np.full(int(m.sum()), t["label"]))
        p.append(preds[i][m])
        v = e56.firmware_vote(preds[i])
        vy.append(np.full(len(v), t["label"]))
        vp.append(v)
    y, p = np.concatenate(y), np.concatenate(p)
    return {"n_frames": int(len(y)), "acc": float((y == p).mean() * 100),
            "vote_acc": float((np.concatenate(vy) == np.concatenate(vp)).mean() * 100)}


def centred_fraction(X: np.ndarray) -> float:
    """สัดส่วนเฟรมที่จุดศูนย์กลางมืออยู่ใน (3, 4] ทั้งแถวและคอลัมน์ (ตรวจข้อ ค.)"""
    hand = X[..., 0] < rc.HAND_THRESH
    cnt = hand.sum(axis=(1, 2))
    ok = cnt > 0
    r = (hand * np.arange(8)[None, :, None]).sum(axis=(1, 2)) / np.maximum(cnt, 1)
    c = (hand * np.arange(8)[None, None, :]).sum(axis=(1, 2)) / np.maximum(cnt, 1)
    inbox = ok & (r > 3) & (r <= 4) & (c > 3) & (c <= 4)
    return float(inbox.sum() / max(int(ok.sum()), 1))


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/phase56"))
    ap.add_argument("--ref-dir", type=Path, default=Path("data/phase4"), help="ตำแหน่งมือปกติ (รายงานเท่านั้น)")
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--e5r", type=Path, default=Path("results_phase55c_final/deploy/E5R.keras"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_a15"))
    args = ap.parse_args(argv)

    # ---- ห้ามแตะชุดยืนหรือชุดอื่น (§5.1y) ----
    name = str(args.data_dir).replace("\\", "/").rstrip("/").split("/")[-1].lower()
    if name != ALLOWED_DATA_DIR:
        raise SystemExit(f"--data-dir {args.data_dir}: A15 chooses on data\\phase56 ONLY. "
                         "data\\phase56_stand stays untouched until the decision is recorded (§5.1y).")
    for f in (args.e5, args.e5r):
        if not f.is_file():
            raise SystemExit(f"model not found: {f}")
    sha_e5, sha_e5r = e56.sha256_file(args.e5), e56.sha256_file(args.e5r)
    out = args.out_dir
    prev = out / "decision.json"
    if prev.is_file():
        old = json.loads(prev.read_text(encoding="utf-8"))
        if old.get("e5r_sha256") != sha_e5r:
            raise SystemExit(f"{prev} already holds a decision for a DIFFERENT E5R file "
                             f"({old.get('e5r_sha256', '?')[:16]}…) — one decision per model (§5.1y). Stop and ask.")
        print(f"  note: {prev} exists for this same E5R file — re-running to confirm the same result")

    st = e56.import_st_dataset()
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            if mdl.output_shape[-1] != len(CLASSES):
                raise SystemExit(f"{path}: {mdl.output_shape[-1]} outputs, expected 7")
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)

    print(f"Loading {args.data_dir} ...")
    takes = e56.load_takes(args.data_dir, st)
    users = sorted({t["user"] for t in takes})
    if not all(t["board"]["present"] for t in takes):
        raise SystemExit("some takes have no *_board.csv — E5's on-board answers are required (§5.1y rule 2)")
    print(f"  {len(takes)} takes, users {users}, frames {sum(t['n'] for t in takes)}")

    # ---- E5: ผลของบอร์ด (ใช้ตัดสิน) + คำนวณใหม่บน PC (ใช้ตรวจตัวเอง ข้อ ก.) ----
    board_cls = {i: np.where(t["board"]["cls"] == -9, -1, t["board"]["cls"]) for i, t in enumerate(takes)}
    print("Predicting with E5 on the PC (self-check against the board's own answers) ...")
    e5_cls, e5_prob = e56.predict_takes(takes, load_model(args.e5))
    bvp = e56.board_vs_pc(takes, e5_cls, e5_prob)
    if not (bvp["compared"] > 0 and bvp["agree"] == bvp["compared"] and bvp["gate_mismatch"] == 0
            and bvp["missing_rows"] == 0):
        raise SystemExit(f"self-check A failed — E5 on PC != E5 on board ({bvp['agree']}/{bvp['compared']}, "
                         f"gate {bvp['gate_mismatch']}, missing {bvp['missing_rows']}). Evaluation pipeline differs "
                         "from 26 Oct; stop before deciding anything.")
    e5b = pooled(takes, board_cls)
    if abs(e5b["acc"] - E5_BOARD_EXPECTED) > E5_BOARD_TOL:
        raise SystemExit(f"self-check B failed — E5 on board = {e5b['acc']:.2f} %, FACTS §5.1p says "
                         f"{E5_BOARD_EXPECTED} %. Wrong data folder? Stop.")
    print(f"  self-checks A+B PASS — E5 board = PC on {bvp['agree']:,}/{bvp['compared']:,} frames; "
          f"E5 on board {e5b['acc']:.1f} % (= §5.1p)")

    # ---- E5R: PC, ข้อมูล re-centre แล้ว ----
    X_all = np.concatenate([t["X"] for t in takes])
    cf = centred_fraction(rc.recentre(X_all))
    if cf < 0.99:
        raise SystemExit(f"self-check C failed — only {cf:.2%} of frames centred after recentre()")
    print(f"  self-check C PASS — after recentre() {cf * 100:.2f} % of {len(X_all):,} frames are centred")
    print("Predicting with E5R on the PC (re-centred input) ...")
    m_r = load_model(args.e5r)
    r_cls, _ = e56.predict_takes(takes, lambda X: m_r(rc.recentre(X)))
    e5r = pooled(takes, r_cls)

    # ---- กติกา ----
    thr = e5b["acc"] - MARGIN_PP
    bug = e5r["acc"] < BUG_GUARD_PCT
    passes = (not bug) and e5r["acc"] >= thr
    decision = ("STOP_BUG_GUARD" if bug else "E5R_PASSES_A15" if passes else "E5_STAYS")

    # ---- รายงานรายคน / รายท่า / รายรอบ (รายงานเท่านั้น) ----
    per_user = {u: {"E5_board": e56.score(takes, board_cls, u), "E5R": e56.score(takes, r_cls, u)} for u in users}
    ref = [t for t in e56.load_takes(args.ref_dir, st, with_board=False) if t["user"] in users] \
        if args.ref_dir.is_dir() else []
    natural = {}
    for u in users:
        for c in CLASSES:
            hs = [t["hand"] for t in ref if t["user"] == u and t["cls"] == c and len(t["hand"])]
            natural[(u, c)] = e56.centroid(hs) if hs else None
    rows = []
    for i, t in enumerate(takes):
        m = t["mask"]
        a_b = float((board_cls[i][m] == t["label"]).mean() * 100)
        a_r = float((r_cls[i][m] == t["label"]).mean() * 100)
        cen = e56.centroid([t["hand"]]) if len(t["hand"]) else None
        nat = natural.get((t["user"], t["cls"]))
        d = (cen[0] - nat[0], cen[1] - nat[1]) if (cen and nat) else (float("nan"), float("nan"))
        rows.append({"user": t["user"], "class": t["cls"], "take": t["take"], "frames": int(m.sum()),
                     "E5_board_acc": round(a_b, 2), "E5R_acc": round(a_r, 2), "diff_pp": round(a_r - a_b, 2),
                     "centroid_row": round(cen[0], 2) if cen else "", "centroid_col": round(cen[1], 2) if cen else "",
                     "d_row_vs_phase4": round(d[0], 2), "d_col_vs_phase4": round(d[1], 2),
                     "file": t["file"].name})
    diffs = [r["diff_pp"] for r in rows]
    wil = {"n_takes": len(diffs), "n_nonzero": int(sum(abs(x) > 1e-9 for x in diffs))}
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs)
        wil.update({"W": float(res.statistic), "p": float(res.pvalue)})
    except (ImportError, ValueError) as exc:          # เช่น ต่างกันเป็นศูนย์ทุกรอบ
        wil["note"] = str(exc)

    # ---- เขียนผล ----
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_take.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stamp = datetime.now().isoformat(timespec="seconds")
    dec = {"decided_at": stamp, "rule": "FACTS §5.1y A15 (pre-registered, commit a0a0ed2)",
           "data_dir": str(args.data_dir), "e5_sha256": sha_e5, "e5r_sha256": sha_e5r,
           "E5_board": e5b, "E5R_pc_recentred": e5r, "threshold_pct": thr, "bug_guard_pct": BUG_GUARD_PCT,
           "bug_guard_triggered": bug, "decision": decision, "self_check_board_vs_pc": bvp,
           "centred_fraction": cf, "take_level_wilcoxon_descriptive": wil,
           "still_required_if_pass": ["board = PC 100 % for E5R (MY_TOF_RECENTRE firmware)", "per-frame work <= 33 ms"],
           "phase56_stand": "untouched — score only after this decision is committed (§5.1y rule 5)"}
    (out / "decision.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    L = ["=" * 78, "  A15 decision  ·  E5R vs E5 on data\\phase56 (sitting)  ·  FACTS §5.1y", "=" * 78,
         f"  E5  (ON THE BOARD, recorded 26 Oct)   per-frame {e5b['acc']:6.2f} %   vote {e5b['vote_acc']:6.2f} %"
         f"   n = {e5b['n_frames']}",
         f"  E5R (PC, re-centred input)            per-frame {e5r['acc']:6.2f} %   vote {e5r['vote_acc']:6.2f} %",
         f"  difference E5R - E5                   {e5r['acc'] - e5b['acc']:+6.2f} pp", "",
         f"  Rule 1  bug guard: E5R >= {BUG_GUARD_PCT:.0f} %                 "
         f"{'TRIGGERED -> STOP, look for a bug' if bug else 'ok (not triggered)'}",
         f"  Rule 2  E5R >= E5 - {MARGIN_PP:.1f} = {thr:.2f} %             {'HOLDS' if passes else 'DOES NOT HOLD'}",
         f"  DECISION: {decision}"
         + ("  (still needs board = PC 100 % and <= 33 ms before deployment)" if passes else ""), ""]
    for u in users:
        L.append(f"  ── {u}: recall %  " + "  ".join(f"{c[:6]:>6}" for c in CLASSES))
        for lab, key in (("E5 board", "E5_board"), ("E5R", "E5R")):
            r = per_user[u][key]["recall"]
            L.append(f"    {lab:<14}   " + "  ".join(f"{r[c]:6.1f}" for c in CLASSES))
        L.append("    E5R top confusions: " + (", ".join(f"{a}→{b} {n}" for n, a, b in per_user[u]["E5R"]["top_confusions"])
                                                or "none"))
    L += ["", "  ── per take (report only; d_row/d_col = hand centroid minus this posture's phase4 position) ──",
          f"    {'take':<16}{'E5 board':>9}{'E5R':>8}{'diff':>8}{'d_row':>8}{'d_col':>8}"]
    for r in rows:
        star = "  ← E5 failure take" if r["E5_board_acc"] < 90 else ""
        L.append(f"    {r['class'] + ' ' + str(r['take']):<16}{r['E5_board_acc']:>9.1f}{r['E5R_acc']:>8.1f}"
                 f"{r['diff_pp']:>+8.1f}{r['d_row_vs_phase4']:>+8.2f}{r['d_col_vs_phase4']:>+8.2f}{star}")
    wtxt = (f"W = {wil['W']:.1f}, p = {wil['p']:.3f}" if "p" in wil else wil.get("note", ""))
    L += [f"    take-level paired Wilcoxon (descriptive, {wil['n_nonzero']} of {wil['n_takes']} takes differ): {wtxt}", "",
          "  Pre-registered predictions (§5.1y)",
          f"    {'HOLDS' if not bug else 'FAILS'}  N1 bug guard not triggered",
          f"    {'HOLDS' if e5r['acc'] >= 90.0 else 'FAILS'}  N2 E5R on phase56 >= 90.0 %   [{e5r['acc']:.2f}]", "",
          "  Self-checks: A board = PC for E5 on every frame · B E5 board = 93.0 % (§5.1p) · C input centred — all PASS",
          "  n = 1 subject, 1 sitting session, 2 takes per posture: a known-user safety check, not proof of 'better'.",
          "  data\\phase56_stand was NOT opened. Commit results_a15\\ before scoring it.",
          f"  output: {out}", "=" * 78]
    text = "\n".join(L)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return dec


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ชี้ --data-dir ไปชุดอื่น (โดยเฉพาะ phase56_stand) → หยุดทันที (ชุดยืนต้องไม่ถูกแตะก่อนตัดสิน)
#  2. ตรวจตัวเอง ก/ข/ค ไม่ผ่าน → หยุดก่อนตัดสิน แปลว่าการอ่าน/เตรียมข้อมูลต่างจากวันที่ 26 ต.ค. หรือ re-centre ไม่ทำงาน
#  3. ตัวกันบั๊กทำงาน (< 80 %) → ไม่ใช่ "E5R แพ้" แต่ต้องหาบั๊กก่อน (เช่น ลืม re-centre, ลำดับคลาส)
#  4. ห้ามเปลี่ยนเกณฑ์ 3.0 pp / 80 % หลังเห็นผล — ล็อกไว้ใน FACTS §5.1y ตั้งแต่ commit a0a0ed2
#  5. ผลรายท่า/รายรอบ "ไม่ใช่" เกณฑ์ (ท่าละ 2 รอบ) — ใช้อธิบายเท่านั้น
#  6. ถ้า E5R.keras ถูกสร้างใหม่ (sha256 เปลี่ยน) แต่ results_a15\decision.json ยังอยู่ → หยุด ห้ามตัดสินซ้ำด้วยโมเดลใหม่
# ─────────────────────────────────────────────────────────────────────────
