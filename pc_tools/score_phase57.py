"""score_phase57.py — ให้คะแนนชุดยืนยันสุดท้าย data\\phase57 (User 6 วันใหม่ ยืน, E5R บนบอร์ด) ครั้งเดียว (PROJECT_FACTS §5.1y-W)

ชุดนี้ "ไม่เคยแตะ" — เก็บหลังเลือกโมเดลเสร็จแล้ว จึงเป็นตัวเลขที่ไม่ลำเอียงตัวเดียวของคนที่โมเดลรู้จัก วันใหม่ บนบอร์ดจริง
รายงานอย่างเดียว: ผลไม่ว่าออกมาเท่าไหร่ ห้ามเปลี่ยนโมเดล ห้ามเทรนด้วยชุดนี้ ห้ามคัด take ทิ้งหลังเห็นผล
  1. ความแม่นยำ "บนบอร์ด" = บรรทัด Y ที่บอร์ดทายสดตอนเก็บ (*_board.csv) เทียบท่าที่ทำจริง — ต่อเฟรม, หลังโหวต 15 เฟรม
     (กติกาเฟิร์มแวร์), ต่อท่า, ต่อ take
  2. ตรวจซ้ำ: เฟรมดิบเดียวกัน → st_dataset → recentre → E5R.keras บน PC ต้องทายตรงกับบอร์ด "ทุกเฟรม"
  3. เวลา: บรรทัด Q ของทุกเฟรมที่ทาย → mean ± SD, max (≤ 33 ms)
  ตัวกันบั๊ก: ถ้าบอร์ด ≠ PC เฟรมใดเฟรมหนึ่ง หรือความแม่นยำ < 80 % → หยุดหาสาเหตุ (เฟิร์มแวร์/สวิตช์/ไฟล์) ก่อนรายงาน

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\score_phase57.py
ค่าเริ่มต้น: --data-dir data\\phase57 --user User6 --keras results_phase55c_final\\deploy\\E5R.keras --out-dir results_phase57
ผลลัพธ์: results_phase57\\  report.txt, summary.json, per_take.csv
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
import evaluate_phase56 as e56     # noqa: E402  (load_takes / predict_takes / score / firmware_vote / sha256_file)
import recentre as rc               # noqa: E402

CLASSES = e56.CLASSES
BUG_GUARD_PCT = 80.0
BUDGET_US = 33000
PROB_TOL = 2e-5


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/phase57"))
    ap.add_argument("--user", default="User6")
    ap.add_argument("--keras", type=Path, default=Path("results_phase55c_final/deploy/E5R.keras"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase57"))
    args = ap.parse_args(argv)
    if not args.keras.is_file():
        raise SystemExit(f"model not found: {args.keras}")
    prev = args.out_dir / "summary.json"
    if prev.is_file():
        old = json.loads(prev.read_text(encoding="utf-8"))
        print(f"  note: {prev} exists (scored {old.get('generated')}) — re-running gives the same numbers; the first score is the result")

    st = e56.import_st_dataset()
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)
    print(f"Loading {args.data_dir} ...")
    takes = e56.load_takes(args.data_dir, st)
    users = sorted({t["user"] for t in takes})
    if users != [args.user]:
        raise SystemExit(f"{args.data_dir}: expected only {args.user}, found {users}")
    if not all(t["board"]["present"] for t in takes):
        raise SystemExit("some takes have no *_board.csv — the board's own answers are the result; re-collect with collect_own_data v6+")
    per_cls = {c: sum(t["cls"] == c for t in takes) for c in CLASSES}
    print(f"  {len(takes)} takes ({', '.join(f'{c} {n}' for c, n in per_cls.items())}), {sum(t['n'] for t in takes)} frames")

    board = {i: np.where(t["board"]["cls"] == -9, -1, t["board"]["cls"]) for i, t in enumerate(takes)}
    missing_y = int(sum((t["board"]["cls"] == -9).sum() for t in takes))
    m = load_model(args.keras)
    pc_cls, pc_prob = e56.predict_takes(takes, lambda X: m(rc.recentre(X)))
    agree = n_all = 0
    pd_max = 0.0
    for i, t in enumerate(takes):
        ok = t["board"]["cls"] != -9
        agree += int((board[i][ok] == pc_cls[i][ok]).sum())
        n_all += int(ok.sum())
        bp = t["board"]["p"][t["mask"]]
        good = np.isfinite(bp).all(axis=1)
        if good.any():
            pd_max = max(pd_max, float(np.max(np.abs(bp[good] - pc_prob[i][good]))))
    parity = agree == n_all and missing_y == 0

    sc = e56.score(takes, board, args.user)
    rows = []
    for i, t in enumerate(takes):
        msk = t["mask"]
        v = e56.firmware_vote(board[i])
        rows.append({"class": t["cls"], "take": t["take"], "frames_in_range": int(msk.sum()),
                     "board_acc": round(float((board[i][msk] == t["label"]).mean() * 100), 2) if msk.any() else "",
                     "vote_acc": round(float((v == t["label"]).mean() * 100), 2) if len(v) else "",
                     "top_wrong": (lambda w: CLASSES[int(np.bincount(w, minlength=7).argmax())] if len(w) else "")(
                         board[i][msk][(board[i][msk] != t["label"]) & (board[i][msk] >= 0)]),
                     "file": t["file"].name})
    times = np.concatenate([t["board"]["t"][:, 4][np.isfinite(t["board"]["t"][:, 4])] for t in takes])
    guard = sc["acc"] < BUG_GUARD_PCT or not parity
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_take.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "rule": "FACTS §5.1y-W (report only)",
               "data_dir": str(args.data_dir), "user": args.user, "keras": str(args.keras), "keras_sha256": e56.sha256_file(args.keras),
               "takes": len(takes), "frames_in_range": sc["n_frames"], "board_acc": sc["acc"], "board_vote_acc": sc.get("vote_acc"),
               "recall": sc["recall"], "top_confusions": sc["top_confusions"], "parity_board_eq_pc": parity,
               "parity_agree": agree, "parity_frames": n_all, "missing_y_rows": missing_y, "prob_maxdiff": pd_max,
               "timing_total_us": {"n": int(len(times)), "mean": float(times.mean()) if len(times) else None,
                                   "sd": float(times.std(ddof=1)) if len(times) > 1 else None,
                                   "max": float(times.max()) if len(times) else None},
               "bug_guard_triggered": guard}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    L = ["=" * 84, f"  Phase 5.7 confirmation · {args.user} new day, standing · E5R on the board · FACTS §5.1y-W (report only)", "=" * 84,
         f"  takes {len(takes)} · in-range frames {sc['n_frames']:,}",
         f"  BOARD accuracy per frame {sc['acc']:6.2f} %   after 15-frame vote {sc.get('vote_acc', float('nan')):6.2f} %",
         "  recall %  " + "  ".join(f"{c[:6]:>6}" for c in CLASSES),
         "            " + "  ".join(f"{sc['recall'][c]:6.1f}" for c in CLASSES),
         "  top confusions: " + (", ".join(f"{a}→{b} {n}" for n, a, b in sc["top_confusions"]) or "none"),
         f"  board = PC (st_dataset → recentre → {args.keras.name}) on every frame: {'YES' if parity else 'NO'} "
         f"({agree:,}/{n_all:,}, missing Y rows {missing_y}, max p diff {pd_max:.1e})"]
    if len(times):
        L.append(f"  per-frame work {times.mean() / 1000:.3f} ± {times.std(ddof=1) / 1000:.3f} ms, max {times.max() / 1000:.3f} ms "
                 f"(n = {len(times):,}; budget {BUDGET_US / 1000:.0f} ms)")
    L += ["", "  per take:        board %   vote %   most frequent wrong answer"]
    for r in rows:
        L.append(f"    {r['class'] + ' ' + str(r['take']):<15}{r['board_acc']:>8}{r['vote_acc']:>9}   {r['top_wrong']}")
    L += ["", ("  BUG GUARD TRIGGERED — find the cause (firmware switches, model files, parity) before reporting any number"
               if guard else "  bug guard not triggered · report these numbers as measured (n = 1 subject, 1 session, 2 takes/posture)"),
          f"  output: {out}", "=" * 84]
    text = "\n".join(L)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return summary


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. เก็บโดยไม่มีไฟล์ _board.csv (collector รุ่นเก่า) → หยุด ; ใช้ collect_own_data.py v6.2 เท่านั้น
#  2. ชุดนี้ต้องมีแต่ User6 — ถ้ามีคนอื่นปน → หยุด
#  3. บอร์ด ≠ PC หรือ < 80 % → ตัวกันบั๊กทำงาน: หาสาเหตุก่อน (เช่น ลืมแฟลช E5R, สวิตช์ REPLAY ค้าง)
#  4. ห้ามใช้ชุดนี้เทรนหรือเลือกอะไร (train scripts ปฏิเสธชื่อ phase57 อยู่แล้ว)
# ─────────────────────────────────────────────────────────────────────────
