"""a17_decide.py — กติกา A17 (PROJECT_FACTS §5.1y-S): ตัดสินครั้งสุดท้ายว่า E5R ขึ้นแทน E5 ได้ไหม โดยยึด "ท่ายืน" เป็นหลัก
ใช้ชุดยืน data\\phase56_stand เท่านั้น — เขียนกติกาและ commit แล้ว (e7a803d) ก่อน E5R เคยถูกวัดบนชุดนี้

กติกา A17 (ต้องผ่าน "ทุกข้อ"):
  1. คนใหม่ (LOSO) ≥ 86.30 %       — วัดแล้ว 89.67 % (§5.1x) → ผ่าน (สคริปต์แค่บันทึก)
  2. ท่ายืนรวม  : E5R (PC, re-centre) ≥ E5 "ที่บอร์ดทายเอง" (บรรทัด Y ใน *_board.csv) บนเฟรมเดียวกัน
  3. ท่ายืนทุกท่า: ไม่มีท่าไหนที่ recall ของ E5R ต่ำกว่า E5 (บอร์ด) เกิน 5.0 pp
  4. มือเลื่อน   : A6-sim แถว −1 (โค้ดเดียวกับ a6_sim.py) — E5R ≥ E5 ; รายงานทุกการเลื่อน
                  (ข้อนี้ re-centre แก้ให้เกือบทั้งหมดอยู่แล้วโดยวิธีการ = เช็คความเรียบร้อย ไม่ใช่หลักฐานการเลื่อนจริง)
  5. บอร์ด      : ทำทีหลัง (X-CUBE-AI + เฟิร์มแวร์ MY_TOF_RECENTRE) — สคริปต์นี้แค่เตือน
ไม่ผ่านข้อใดข้อหนึ่ง → E5 อยู่บนบอร์ดต่อ และ "จบการหาโมเดล"

ตรวจตัวเองก่อนอ่านกติกา (ไม่ผ่าน = หยุด ห้ามตัดสิน):
  ก. E5 บน PC = ผลของบอร์ดทุกเฟรม (§5.1u: 4,221/4,221)
  ข. E5 บนบอร์ดได้ 98.7 % ตาม §5.1u (คลาดได้ ≤ 0.05 pp)
  ค. ข้อมูลที่ส่งให้ E5R ถูก re-centre จริง (≥ 99 % ของเฟรม)
  ง. A6-sim ของ E5 ได้เท่าเดิม: เทียบกับ summary.json ของรอบ 26 ต.ค. แบบตรงเป๊ะ (ถ้ามีไฟล์)
     ถ้าไม่มีไฟล์ → เทียบกับ §5.1v (ไม่เลื่อน 98.7 %, แถว −1 58.3 %, คลาด ≤ 0.05 pp) → ยืนยันว่าจำลองแบบเดียวกัน
  จ. ไฟล์ E5R ต้องเป็นไฟล์เดียวกับที่ตัดสิน A15 (sha256 ตรงกับ results_a15\\decision.json)

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a17_decide.py
ค่าเริ่มต้น: --data-dir data\\phase56_stand --e5 results_phase55_final\\deploy\\E5.keras
            --e5r results_phase55c_final\\deploy\\E5R.keras --a15-decision results_a15\\decision.json --out-dir results_a17
ผลลัพธ์: results_a17\\  decision.json, report.txt, per_take.csv, a6sim_table.csv
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
import evaluate_phase56 as e56     # noqa: E402  (load_takes / predict_takes / score / board_vs_pc / sha256_file)
import recentre as rc               # noqa: E402
import a6_sim                       # noqa: E402  (prepare / predict_all / metrics — โค้ดเดียวกับ §5.1v)
from a15_decide import centred_fraction, pooled   # noqa: E402  (ฟังก์ชันเดียวกับ A15)

CLASSES = e56.CLASSES
ALLOWED_DATA_DIR = "phase56_stand"  # §5.1y-S: ชุดยืนเท่านั้น
LOSO_E5R, LOSO_THRESHOLD = 89.67, 86.30          # §5.1x / A16 ข้อ 1 (วัดแล้ว — บันทึกเท่านั้น)
POSTURE_MARGIN_PP = 5.0                          # A17 ข้อ 3
E5_BOARD_EXPECTED = 98.7                         # §5.1u
E5_SIM_NONE_EXPECTED, E5_SIM_ROWM1_EXPECTED = 98.7, 58.3   # §5.1v standing
TOL = 0.05
RULE4_SHIFT = "row-1"


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/phase56_stand"))
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--e5r", type=Path, default=Path("results_phase55c_final/deploy/E5R.keras"))
    ap.add_argument("--a15-decision", type=Path, default=Path("results_a15/decision.json"))
    ap.add_argument("--a6sim-summary", type=Path,
                    default=Path("results_a6sim/eval_20261026_221509/summary.json"), help="ผล A6-sim ของ E5 (§5.1v)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_a17"))
    args = ap.parse_args(argv)

    # ---- ใช้ชุดยืนเท่านั้น ----
    name = str(args.data_dir).replace("\\", "/").rstrip("/").split("/")[-1].lower()
    if name != ALLOWED_DATA_DIR:
        raise SystemExit(f"--data-dir {args.data_dir}: A17 decides on data\\phase56_stand ONLY (§5.1y-S).")
    for f in (args.e5, args.e5r, args.a15_decision):
        if not f.is_file():
            raise SystemExit(f"not found: {f}")
    sha_e5, sha_e5r = e56.sha256_file(args.e5), e56.sha256_file(args.e5r)

    # ---- ตรวจ จ.: E5R ต้องเป็นไฟล์เดียวกับ A15 (ห้ามเปลี่ยนโมเดลหลังเห็นผล) ----
    a15 = json.loads(args.a15_decision.read_text(encoding="utf-8"))
    if a15.get("e5r_sha256") != sha_e5r or a15.get("e5_sha256") != sha_e5:
        raise SystemExit("self-check E failed — E5/E5R files differ from the ones scored in A15 "
                         f"({args.a15_decision}). A17 is for the SAME E5R file only (§5.1y-S). Stop.")
    out = args.out_dir
    prev = out / "decision.json"
    if prev.is_file():
        old = json.loads(prev.read_text(encoding="utf-8"))
        if old.get("e5r_sha256") != sha_e5r:
            raise SystemExit(f"{prev} already holds a decision for a DIFFERENT E5R file — one decision only. Stop and ask.")
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
    if len(users) != 1:
        raise SystemExit(f"one subject expected in {args.data_dir}, found {users}")
    if not all(t["board"]["present"] for t in takes):
        raise SystemExit("some takes have no *_board.csv — E5's on-board answers are required")
    print(f"  {len(takes)} takes, user {users[0]}, frames {sum(t['n'] for t in takes)}")

    # ---- ตรวจ ก. + ข. : E5 ----
    board_cls = {i: np.where(t["board"]["cls"] == -9, -1, t["board"]["cls"]) for i, t in enumerate(takes)}
    print("Predicting with E5 on the PC (self-check against the board) ...")
    m_e5 = load_model(args.e5)
    e5_cls, e5_prob = e56.predict_takes(takes, m_e5)
    bvp = e56.board_vs_pc(takes, e5_cls, e5_prob)
    if not (bvp["compared"] > 0 and bvp["agree"] == bvp["compared"] and bvp["gate_mismatch"] == 0
            and bvp["missing_rows"] == 0):
        raise SystemExit(f"self-check A failed — E5 PC != board ({bvp['agree']}/{bvp['compared']}, gate "
                         f"{bvp['gate_mismatch']}, missing {bvp['missing_rows']}). Stop before deciding.")
    e5b = pooled(takes, board_cls)
    if abs(e5b["acc"] - E5_BOARD_EXPECTED) > TOL:
        raise SystemExit(f"self-check B failed — E5 on board {e5b['acc']:.2f} %, §5.1u says {E5_BOARD_EXPECTED} %. Stop.")
    print(f"  self-checks A+B PASS — board = PC on {bvp['agree']:,}/{bvp['compared']:,} frames; E5 board {e5b['acc']:.2f} %")

    # ---- ตรวจ ค. ----
    X_all = np.concatenate([t["X"] for t in takes])
    cf = centred_fraction(rc.recentre(X_all))
    if cf < 0.99:
        raise SystemExit(f"self-check C failed — only {cf:.2%} of frames centred after recentre()")
    print(f"  self-check C PASS — {cf * 100:.2f} % of {len(X_all):,} frames centred after recentre()")

    # ---- A6-sim (ตรวจ ง. ด้วย E5 ก่อน แล้วค่อยใช้กับ E5R) ----
    print("A6-sim: preparing shifted frames (same code as a6_sim.py) ...")
    for t in takes:
        with np.load(t["file"], allow_pickle=True) as z:
            t["zone"] = z["zone_data"].astype(np.float64)
    prep = a6_sim.prepare(takes, st)
    seq_e5 = a6_sim.predict_all(prep, takes, m_e5)
    sim_e5 = {s: a6_sim.metrics(takes, seq_e5[s], seq_e5["none"]) for s in a6_sim.SHIFTS}
    ref_none, ref_m1, tol, src = E5_SIM_NONE_EXPECTED, E5_SIM_ROWM1_EXPECTED, TOL, "§5.1v (rounded)"
    if args.a6sim_summary.is_file():                      # ใช้ตัวเลขเต็มจากรอบ 26 ต.ค. ถ้ามี → ต้องตรงเป๊ะ
        ref = json.loads(args.a6sim_summary.read_text(encoding="utf-8"))
        ds = [d for d in ref["datasets"].values() if d["name"] == ALLOWED_DATA_DIR]
        if len(ds) != 1:
            raise SystemExit(f"{args.a6sim_summary}: no single '{ALLOWED_DATA_DIR}' entry")
        ref_none, ref_m1 = ds[0]["models"]["E5"]["none"]["acc"], ds[0]["models"]["E5"][RULE4_SHIFT]["acc"]
        tol, src = 1e-6, str(args.a6sim_summary)
    if abs(sim_e5["none"]["acc"] - ref_none) > tol or abs(sim_e5[RULE4_SHIFT]["acc"] - ref_m1) > tol:
        raise SystemExit(f"self-check D failed — E5 A6-sim none {sim_e5['none']['acc']:.4f} / row-1 "
                         f"{sim_e5[RULE4_SHIFT]['acc']:.4f} %, reference ({src}) {ref_none:.4f} / {ref_m1:.4f}. Stop.")
    print(f"  self-check D PASS — E5 A6-sim none {sim_e5['none']['acc']:.2f} %, row-1 {sim_e5[RULE4_SHIFT]['acc']:.2f} % "
          f"(= {src})")

    # ---- E5R ----
    print("Predicting with E5R on the PC (re-centred input) ...")
    m_r = load_model(args.e5r)
    pred_r = lambda X: m_r(rc.recentre(X))      # noqa: E731  — re-centre ทุกครั้งก่อนเข้า E5R
    r_cls, _ = e56.predict_takes(takes, pred_r)
    e5r = pooled(takes, r_cls)
    seq_r = a6_sim.predict_all(prep, takes, pred_r)
    sim_r = {s: a6_sim.metrics(takes, seq_r[s], seq_r["none"]) for s in a6_sim.SHIFTS}
    if abs(sim_r["none"]["acc"] - e5r["acc"]) > 1e-6:
        raise SystemExit(f"internal check failed — E5R unshifted A6-sim {sim_r['none']['acc']:.4f} != direct {e5r['acc']:.4f}")

    # ---- กติกา ----
    u = users[0]
    rec_b = e56.score(takes, board_cls, u)["recall"]
    rec_r = e56.score(takes, r_cls, u)["recall"]
    worst = {c: rec_r[c] - rec_b[c] for c in CLASSES if not np.isnan(rec_b[c])}
    rules = {
        "1_loso": {"value": LOSO_E5R, "threshold": LOSO_THRESHOLD, "holds": LOSO_E5R >= LOSO_THRESHOLD,
                   "note": "measured earlier (§5.1x), recorded only"},
        "2_standing_overall": {"E5R": e5r["acc"], "E5_board": e5b["acc"], "holds": e5r["acc"] >= e5b["acc"]},
        "3_every_posture": {"diff_pp": worst, "margin_pp": POSTURE_MARGIN_PP,
                            "holds": all(d >= -POSTURE_MARGIN_PP for d in worst.values())},
        "4_shift_row_minus_1": {"E5R": sim_r[RULE4_SHIFT]["acc"], "E5": sim_e5[RULE4_SHIFT]["acc"],
                                "holds": sim_r[RULE4_SHIFT]["acc"] >= sim_e5[RULE4_SHIFT]["acc"]},
    }
    passes = all(r["holds"] for r in rules.values())
    decision = "E5R_PASSES_A17_RULES_1_TO_4" if passes else "E5_STAYS_SEARCH_ENDS"

    # ---- รายรอบ (รายงานเท่านั้น) ----
    rows = []
    for i, t in enumerate(takes):
        m = t["mask"]
        a_b = float((board_cls[i][m] == t["label"]).mean() * 100)
        a_r = float((r_cls[i][m] == t["label"]).mean() * 100)
        rows.append({"class": t["cls"], "take": t["take"], "frames": int(m.sum()), "E5_board_acc": round(a_b, 2),
                     "E5R_acc": round(a_r, 2), "diff_pp": round(a_r - a_b, 2), "file": t["file"].name})

    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_take.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "a6sim_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "shift", "acc", "vote_acc"] + [f"recall_{c}" for c in CLASSES])
        for mname, sim in (("E5", sim_e5), ("E5R", sim_r)):
            for s, r in sim.items():
                w.writerow([mname, s, f"{r['acc']:.2f}", f"{r['vote_acc']:.2f}"] + [f"{r['recall'][c]:.2f}" for c in CLASSES])
    dec = {"decided_at": datetime.now().isoformat(timespec="seconds"),
           "rule": "FACTS §5.1y-S A17 (pre-registered, commit e7a803d)", "data_dir": str(args.data_dir),
           "e5_sha256": sha_e5, "e5r_sha256": sha_e5r, "E5_board": e5b, "E5R_pc_recentred": e5r,
           "recall_E5_board": rec_b, "recall_E5R": rec_r, "rules": rules, "decision": decision,
           "a6sim_acc": {"E5": {s: r["acc"] for s, r in sim_e5.items()}, "E5R": {s: r["acc"] for s, r in sim_r.items()}},
           "self_checks": {"A_board_vs_pc": bvp, "B_e5_board": e5b["acc"], "C_centred_fraction": cf,
                           "D_e5_sim": [sim_e5["none"]["acc"], sim_e5[RULE4_SHIFT]["acc"]], "D_reference": src, "E_same_files_as_A15": True},
           "still_required_if_pass": ["rule 5: X-CUBE-AI + MY_TOF_RECENTRE, board = PC 100 % (replay + live), <= 33 ms"],
           "caveat": "phase56_stand is now the selection set → the deployed model's standing figure is slightly optimistic"}
    (out / "decision.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    yes = lambda b: "HOLDS" if b else "FAILS"   # noqa: E731
    r3 = rules["3_every_posture"]
    L = ["=" * 84, "  A17 decision · E5R vs E5 on data\\phase56_stand (standing) · FACTS §5.1y-S", "=" * 84,
         f"  E5  (ON THE BOARD)          per-frame {e5b['acc']:6.2f} %   vote {e5b['vote_acc']:6.2f} %   n = {e5b['n_frames']}",
         f"  E5R (PC, re-centred input)  per-frame {e5r['acc']:6.2f} %   vote {e5r['vote_acc']:6.2f} %", "",
         f"  Rule 1  LOSO {LOSO_E5R:.2f} >= {LOSO_THRESHOLD:.2f} %  (measured §5.1x)          {yes(rules['1_loso']['holds'])}",
         f"  Rule 2  standing overall E5R >= E5 board ({e5r['acc']:.2f} vs {e5b['acc']:.2f})        "
         f"{yes(rules['2_standing_overall']['holds'])}",
         f"  Rule 3  no posture more than {POSTURE_MARGIN_PP:.1f} pp below E5                      {yes(r3['holds'])}",
         f"  Rule 4  A6-sim row-1 E5R >= E5 ({sim_r[RULE4_SHIFT]['acc']:.2f} vs {sim_e5[RULE4_SHIFT]['acc']:.2f})       "
         f"{yes(rules['4_shift_row_minus_1']['holds'])}",
         "  Rule 5  board = PC + <= 33 ms                                  (later, after firmware)",
         f"  DECISION: {decision}", "",
         "  recall %      " + "".join(f"{c[:9]:>11}" for c in CLASSES),
         "  E5 board      " + "".join(f"{rec_b[c]:>11.1f}" for c in CLASSES),
         "  E5R           " + "".join(f"{rec_r[c]:>11.1f}" for c in CLASSES),
         "  E5R − E5      " + "".join(f"{worst.get(c, float('nan')):>+11.1f}" for c in CLASSES), "",
         "  A6-sim per-frame %  " + "".join(f"{s:>7}" for s in a6_sim.SHIFTS),
         "  E5                  " + "".join(f"{sim_e5[s]['acc']:>7.1f}" for s in a6_sim.SHIFTS),
         "  E5R                 " + "".join(f"{sim_r[s]['acc']:>7.1f}" for s in a6_sim.SHIFTS),
         "  (A6-sim: re-centring undoes a whole-image shift by construction — sanity check, not physical proof)", "",
         "  per take:      E5 board     E5R    diff"]
    for r in rows:
        L.append(f"    {r['class'] + ' ' + str(r['take']):<14}{r['E5_board_acc']:>8.1f}{r['E5R_acc']:>8.1f}{r['diff_pp']:>+8.1f}")
    L += ["", "  Self-checks A (board = PC) · B (E5 98.7 %) · C (centred) · D (A6-sim = §5.1v) · E (same files as A15): all PASS",
          "  n = 1 subject, 1 standing session, 2 takes per posture. phase56_stand is now the selection set (§5.1y-S).",
          f"  output: {out}", "=" * 84]
    text = "\n".join(L)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return dec


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ชี้ --data-dir ไปชุดอื่น → หยุด (A17 ตัดสินบนชุดยืนเท่านั้น)
#  2. ตรวจ ก–จ ไม่ผ่าน → หยุดก่อนตัดสิน: อ่าน/เตรียมข้อมูลต่างจากวันที่ 26 ต.ค., re-centre ไม่ทำงาน,
#     การจำลองเลื่อนต่างจาก §5.1v หรือไฟล์โมเดลไม่ใช่ตัวเดียวกับ A15 → แก้โค้ดประเมิน แล้วรันใหม่ด้วยไฟล์ E5R เดิม
#  3. ห้ามเปลี่ยนเกณฑ์ (5.0 pp, ≥ E5) หลังเห็นผล — ล็อกใน FACTS §5.1y-S ตั้งแต่ commit e7a803d
#  4. ข้อ 4 แทบผ่านแน่นอนโดยวิธีการ (re-centre) — อย่ารายงานว่าเป็นหลักฐานว่าทนการเลื่อนมือจริง
#  5. ผ่านข้อ 1–4 ยังไม่ใช่ "ขึ้นบอร์ด" — ต้องผ่านข้อ 5 (board = PC 100 %, ≤ 33 ms) ก่อน
# ─────────────────────────────────────────────────────────────────────────
