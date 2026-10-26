"""a5_eval.py — A5: ทดสอบท่า Love ตำแหน่งขอบ/กลาง โดยให้บอร์ดทายเอง (PROJECT_FACTS §5.1q)

กติกาทั้งหมดจดไว้ใน §5.1q ก่อนเก็บข้อมูล — สคริปต์นี้แค่อ่านและนับ ไม่ปรับอะไร
  ลำดับตำแหน่ง (random.seed(55)):  take 1..10 = centre, edge, centre, centre, edge, centre, centre, edge, edge, edge
  ต่อ 1 รอบ: ใช้ผลทายของ "บอร์ด" ทุกเฟรมที่มือในช่วง (บรรทัด Y ใน *_board.csv)
             ท่าที่บอร์ดทายบ่อยที่สุด = ผลของรอบ, สัดส่วนของท่านั้น = ความนิ่ง
             ผลรอบ = "Love"     ถ้าท่าที่มากที่สุดคือ Love และได้ ≥ 80 % ของเฟรม
                     "unstable" ถ้าท่าที่มากที่สุดได้ < 80 %
                     "<ท่าอื่น>"  ถ้าท่าอื่นได้ ≥ 80 %
  สรุป: centre x/5, edge y/5 (นับเฉพาะ "Love")  เทียบคำทำนาย centre 5/5, edge ≥ 4/5
  เพิ่มเติม: จุดศูนย์กลางมือ (แถว, คอลัมน์) ต่อรอบ → ยืนยันว่าวางขอบ/กลางต่างกันจริง
             PC ทาย E5.keras ซ้ำทุกเฟรม → บอร์ด = PC ไหม (ปิดได้ด้วย --no-pc-check)
  ข้อจำกัด: n = 1 คน, 5 รอบต่อตำแหน่ง → รายงานเชิงพรรณนา ไม่มีค่า p

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a5_eval.py
ตัวเลือก: --data-dir data\\a5_love --ref-dir data\\phase4 --e5 results_phase55_final\\deploy\\E5.keras --no-pc-check
ผลลัพธ์: results_a5\\eval_<วันเวลา>\\ report.txt, summary.json, per_frame.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluate_phase56 import (CLASSES, ST_ROW_RANGE, board_vs_pc, centroid, firmware_vote, import_st_dataset,
                              load_takes, predict_takes, sha256_file, timing_stats)

ORDER = ["centre", "edge", "centre", "centre", "edge", "centre", "centre", "edge", "edge", "edge"]  # §5.1q
STABLE_SHARE = 0.80          # เกณฑ์ "นิ่ง" ที่ตั้งไว้ก่อนเก็บ (§5.1q)
LOVE = CLASSES.index("Love")
PREDICTION = {"centre": 5, "edge": 4}   # คำทำนายของครู: centre 5/5, edge ≥ 4/5


def judge_take(board_cls: np.ndarray) -> dict:
    """ตัดสิน 1 รอบจากผลทายรายเฟรมของบอร์ด (เฉพาะเฟรมในช่วง: ค่า ≥ 0)"""
    seq = board_cls[board_cls >= 0]
    if len(seq) == 0:
        return {"n_pred": 0, "top": None, "share": 0.0, "love_share": 0.0, "result": "no-data", "counts": {}}
    cnt = Counter(int(c) for c in seq)
    top, top_n = cnt.most_common(1)[0]
    share = top_n / len(seq)
    if share < STABLE_SHARE:
        result = "unstable"
    else:
        result = CLASSES[top]
    return {"n_pred": int(len(seq)), "top": CLASSES[top], "share": share, "love_share": cnt.get(LOVE, 0) / len(seq),
            "result": result, "counts": {CLASSES[k]: v for k, v in sorted(cnt.items())}}


def ref_love_centroid(ref_dir: Path, st, user: str) -> tuple[float, float] | None:
    """centroid ของท่า Love ในชุดฝึกของคนเดียวกัน (data\\phase4) ไว้เทียบ"""
    if not ref_dir.is_dir():
        return None
    try:
        ref = [t for t in load_takes(ref_dir, st, with_board=False) if t["cls"] == "Love" and t["user"] == user]
    except SystemExit:
        return None
    masks = [t["hand"] for t in ref if len(t["hand"])]
    return centroid(masks) if masks else None


def main(argv: list[str] | None = None, load_model=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/a5_love"))
    ap.add_argument("--ref-dir", type=Path, default=Path("data/phase4"))
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--no-pc-check", action="store_true", help="ไม่ต้องให้ PC ทายซ้ำ (ไม่โหลด TensorFlow)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    st = import_st_dataset()
    takes = [t for t in load_takes(args.data_dir, st) if t["cls"] == "Love"]
    if not takes:
        raise SystemExit(f"no Love takes under {args.data_dir}")
    nums = [t["take"] for t in takes]
    dup = [k for k, v in Counter(nums).items() if v > 1]
    if dup:
        raise SystemExit(f"take numbers appear more than once: {dup} — keep one file per take")
    takes.sort(key=lambda t: t["take"])
    users = sorted({t["user"] for t in takes})

    rows = []
    for t in takes:
        if not 1 <= t["take"] <= len(ORDER):
            raise SystemExit(f"{t['file'].name}: take {t['take']} is outside 1..{len(ORDER)}")
        if not t["board"]["present"]:
            raise SystemExit(f"{t['file'].name}: _board.csv missing — collect with collect_own_data.py v6+")
        j = judge_take(t["board"]["cls"])
        vote = firmware_vote(t["board"]["cls"])
        r, c = centroid([t["hand"]]) if len(t["hand"]) else (float("nan"), float("nan"))
        rows.append({"take": t["take"], "position": ORDER[t["take"] - 1], "file": t["file"].name, "n_frames": t["n"],
                     "n_in_range": int(t["mask"].sum()), **j,
                     "vote_love_share": float((vote == LOVE).mean()) if len(vote) else 0.0,
                     "centroid_row": r, "centroid_col": c})

    summary = {}
    for pos in ("centre", "edge"):
        pr = [x for x in rows if x["position"] == pos]
        cen = [centroid([t["hand"]]) for t in takes if ORDER[t["take"] - 1] == pos and len(t["hand"])]
        summary[pos] = {"done": len(pr), "love": sum(x["result"] == "Love" for x in pr),
                        "unstable": sum(x["result"] == "unstable" for x in pr),
                        "other": sum(x["result"] not in ("Love", "unstable") for x in pr),
                        "results": [x["result"] for x in pr],
                        "mean_share_love": float(np.mean([x["love_share"] for x in pr])) if pr else float("nan"),
                        "centroid_row_mean": float(np.mean([c[0] for c in cen])) if cen else float("nan"),
                        "centroid_col_mean": float(np.mean([c[1] for c in cen])) if cen else float("nan"),
                        "centroid_row_sd": float(np.std([c[0] for c in cen], ddof=1)) if len(cen) > 1 else float("nan"),
                        "centroid_col_sd": float(np.std([c[1] for c in cen], ddof=1)) if len(cen) > 1 else float("nan"),
                        "prediction_met": (sum(x["result"] == "Love" for x in pr) >= PREDICTION[pos]) if len(pr) == 5
                        else None}

    pc = None
    if not args.no_pc_check:
        if not args.e5.is_file():
            raise SystemExit(f"{args.e5} not found (or use --no-pc-check)")
        if load_model is None:
            import tensorflow as tf  # โหลดเฉพาะตอนต้องใช้
            load_model = tf.keras.models.load_model
        model = load_model(str(args.e5))
        e5_cls, e5_prob = predict_takes(takes, lambda X: model.predict(X, verbose=0))
        pc = board_vs_pc(takes, e5_cls, e5_prob)
        pc["e5_sha256"] = sha256_file(args.e5)

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "data_dir": str(args.data_dir), "users": users,
           "order": ORDER, "stable_share": STABLE_SHARE, "takes": rows, "summary": summary,
           "ref_love_centroid": {u: ref_love_centroid(args.ref_dir, st, u) for u in users},
           "st_love_row_range": ST_ROW_RANGE["Love"], "board_vs_pc": pc, "timing": timing_stats(takes)}

    out = args.out_dir or Path("results_a5") / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["take", "position", "frame_idx", "in_range", "board_class"])
        for t in takes:
            for i in range(t["n"]):
                bc = int(t["board"]["cls"][i])
                w.writerow([t["take"], ORDER[t["take"] - 1], i, int(t["mask"][i]), CLASSES[bc] if bc >= 0 else bc])
    text = report_text(res)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved → {out}")
    return res


def report_text(res: dict) -> str:
    L = ["A5 — Love edge/centre, predicted ON THE BOARD (E5)  ·  PROJECT_FACTS §5.1q",
         f"data: {res['data_dir']}   users: {', '.join(res['users'])}   generated: {res['generated']}",
         f"rule: per take, board's most frequent class over in-range frames; Love only if share >= "
         f"{res['stable_share']:.0%}, else 'unstable'", "",
         f"{'take':>4} {'pos':<7} {'frames':>6} {'inrange':>7}  {'result':<10} {'top share':>9} {'Love %':>7} "
         f"{'vote Love%':>10}  {'row':>5} {'col':>5}"]
    for x in res["takes"]:
        L.append(f"{x['take']:>4} {x['position']:<7} {x['n_frames']:>6} {x['n_in_range']:>7}  {x['result']:<10} "
                 f"{x['share']:>9.1%} {x['love_share']:>7.1%} {x['vote_love_share']:>10.1%}  "
                 f"{x['centroid_row']:>5.2f} {x['centroid_col']:>5.2f}")
        others = {k: v for k, v in x["counts"].items() if k != "Love"}
        if others:
            L.append(f"{'':>13}non-Love classes (frames): {others}")
    L.append("")
    for pos in ("centre", "edge"):
        s = res["summary"][pos]
        met = {True: "MET", False: "NOT MET", None: "n/a (not 5 takes)"}[s["prediction_met"]]
        need = "5/5" if pos == "centre" else ">= 4/5"
        L.append(f"{pos:<7} Love {s['love']}/{s['done']}  (unstable {s['unstable']}, other {s['other']})   "
                 f"prediction {need}: {met}")
        L.append(f"{'':<8}hand centroid row {s['centroid_row_mean']:.2f} ± {s['centroid_row_sd']:.2f}, "
                 f"col {s['centroid_col_mean']:.2f} ± {s['centroid_col_sd']:.2f}  (mean ± SD over takes)")
    for u, c in res["ref_love_centroid"].items():
        L.append(f"reference: {u} Love in training data (phase4) centroid "
                 + (f"row {c[0]:.2f}, col {c[1]:.2f}" if c else "not available"))
    lo, hi = res["st_love_row_range"]
    L.append(f"reference: ST subjects' Love centroid rows {lo:.2f}–{hi:.2f}")
    L.append("reference: old deployed model (24–25 Sep): edge correct, centre unstable/wrong")
    L.append("")
    pc = res["board_vs_pc"]
    if pc:
        L.append(f"board vs PC (E5.keras): {pc['agree']}/{pc['compared']} frames agree, gate mismatches "
                 f"{pc['gate_mismatch']}, missing rows {pc['missing_rows']}, max |Δp| {pc['max_abs_diff']:.2e}")
    else:
        L.append("board vs PC: skipped (--no-pc-check)")
    tm = res["timing"]
    if tm["n"]:
        tt = tm["total_us"]
        L.append(f"board per-frame work: n={tm['n']}, mean {tt['mean']:.0f} µs (SD {tt['sd']:.0f}), "
                 f"max {tt['max']:.0f} µs, over 33 ms: {tm['over_budget']}")
    L.append("limits: n = 1 subject, 5 takes per position, frames within a take are correlated → descriptive only")
    return "\n".join(L)


if __name__ == "__main__":
    main()
